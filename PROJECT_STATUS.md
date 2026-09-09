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
├── calendar/       sessions.py (MarketCalendar wrapper), timezones.py — implemented;
│                   MarketCalendar reaches rules via RuleContext.calendar
├── validators/     base.py (ABC/REGISTRY/register — done)
│                   structural.py (5 rules, 5 done) | temporal.py (4, 2 done)
│                   | numerical.py (4, 4 done) | financial.py (1, 1 done)
├── engine/         runner.py, aggregator.py — both fully implemented
├── reporting/      text_report.py, json_report.py — both fully implemented
└── cli/main.py     typer app, `validate` command — fully wired end-to-end
```

## Implementation status by rule (14 rules total across 4 categories)

> Rule count grew 13 -> 14: `numerical.non_positive_prices` was added as a
> deliberate scope split out of `OhlcRangeViolation` (see Numerical section).
> Implemented: 13 of 14 (structural 5/5, temporal 3/4, numerical 4/4, financial 1/1).

### Structural (`validators/structural.py`) — 5 of 5 implemented
| # | rule_id | Class | Status |
|---|---|---|---|
| 1 | `structural.missing_columns` | `MissingColumns` | ✅ Implemented + fully tested |
| 2 | `structural.invalid_dtypes` | `InvalidDtypes` | ✅ Implemented + fully tested |
| 3 | `structural.unsorted_timestamps` | `UnsortedTimestamps` | ✅ Implemented + fully tested |
| 4 | `structural.duplicate_timestamps` | `DuplicateTimestamps` | ✅ Implemented + fully tested |
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

`DuplicateTimestamps` design decisions (for reference if extending/revisiting):
- **Detection scope: any-position, not consecutive-only.** This was the key design fork.
  `UnsortedTimestamps` uses a `shift(1)` adjacency comparison, but that technique only
  ever compares a row to its immediate predecessor in row order — it would miss two
  rows sharing a timestamp value if they aren't adjacent (e.g. row 2 and row 9 share a
  value but rows 3–8 differ). Because rules run independently of each other (the engine
  provides no ordering/gating guarantee that `UnsortedTimestamps` ran first or passed),
  this rule cannot assume the data is sorted, and therefore cannot rely on "duplicates
  in sorted data are always adjacent" — that assumption only holds if the data is
  already known to be sorted, which this rule can't take for granted. Chose **group_by
  + count** over shift comparison specifically to catch duplicates anywhere in the
  dataset, decoupling this rule's correctness from sort order entirely. Verified via
  `test_fail_non_adjacent_duplicates_detected` and
  `test_fail_unsorted_non_adjacent_duplicates_detected`.
- **Division of labor with `UnsortedTimestamps` is intentional and pre-existing**:
  `UnsortedTimestamps` explicitly excludes ties (`timestamp[i] == timestamp[i-1]`) from
  its own scope, precisely so this rule owns all duplicate-value reporting without
  double-reporting the same condition from two rules.
- Missing `timestamp` column → PASS/skip (`MissingColumns` owns that), same pattern as
  `UnsortedTimestamps`.
- `df.height < 2` → PASS trivially (can't have a duplicate with 0–1 rows).
- Null timestamps are excluded from grouping (`NullValues` owns nulls) — multiple nulls
  must not be reported as duplicates of each other, since a null is an absence, not a
  real repeated value. Verified via `test_pass_multiple_nulls_not_treated_as_duplicates`.
- Row index (`pl.int_range(0, df.height)`) is attached to the DataFrame **before** the
  null filter and the group_by, so reported indices always refer to positions in the
  original (unfiltered) dataset — not positions in some filtered/grouped intermediate.
- Implementation shape: attach index → filter out nulls → `group_by("timestamp").agg(pl.len())`
  → filter to `count > 1` → join back against the indexed/null-filtered frame on
  `timestamp` to recover every row in a duplicated group → `.sort("index")` to restore
  original row order (group_by output order is not guaranteed stable across Polars
  versions, so this sort is required for deterministic output, not cosmetic).
- `affected_rows` = count of **individual rows** participating in any duplicated group
  (e.g. a timestamp appearing 3 times contributes 3 to this count), not the count of
  distinct duplicated timestamp *values*. This mirrors the row-vs-column distinction
  `NullValues` already makes for `affected_rows` vs. `null_counts`.
- `details = {"violations": [{"index", "timestamp", "count"}, ...]}`, capped at
  `config.max_rows_in_details`; `affected_rows` holds the full (uncapped) count. No
  `previous_timestamp`/`current_timestamp` fields here (unlike `UnsortedTimestamps`) —
  duplicates aren't about adjacency, so there's no "previous" row to report; `count`
  instead tells the reader how large the duplicate group is.
- **Severity/status differs from `UnsortedTimestamps` by design**: `default_severity`
  was already set to `CRITICAL` in the stub (vs. `UnsortedTimestamps`'s `WARNING`), and
  status is `FAIL` on violation (vs. `WARN`) — consistent with this codebase's existing
  correlation between `CRITICAL` severity and `FAIL` status (see `MissingColumns`,
  `InvalidDtypes`). Rationale: an out-of-order timestamp is mechanically fixable by
  re-sorting, but a duplicated timestamp is not safely auto-fixable — there's no way to
  know which of the two (or more) rows sharing a timestamp is "the real one" for that
  point in time, so the ambiguity is treated as more severe.
- Message format: `f"{affected_rows} row(s) share {distinct_duplicated_values}
  duplicated timestamp value(s) (first: {first_timestamp}, appearing {first_count}
  time(s))."` — leads with row-level impact then names the first concrete duplicated
  value, consistent with other rules leading with a count + concrete first example.

`NullValues` design decisions (for reference if extending/revisiting):
- Scope: only the 6 `REQUIRED_COLUMNS` (including `timestamp`) are checked; extra columns are ignored, consistent with `MissingColumns`/`InvalidDtypes`.
- If a required column is entirely absent, it's silently skipped (not reported, not treated as "all null") — `MissingColumns` owns absence. Explicit guard: if *no* required columns exist at all, return PASS immediately rather than calling `pl.any_horizontal([])` on an empty expression list — that behavior was never verified against this Polars version, and depending on unverified/undocumented edge-case behavior is a latent break-on-upgrade risk. Short-circuiting on the trivial case avoids the dependency entirely (same category of defensive move as `UnsortedTimestamps`'s `df.height < 2` early return).
- `affected_rows` = count of **distinct rows** with a null in *any* checked column (not sum of per-column counts — a row with 2 nulled columns counts once). Computed via `pl.any_horizontal([pl.col(c).is_null() for c in checked_cols]).sum()` — a single vectorized horizontal-OR across columns, evaluated natively in Polars. Deliberately not done via per-column `pl.arg_where` + Python `set.union()` (Option B, rejected): that approach round-trips indices out of Polars into boxed Python ints for every checked column before combining, which is slower and less idiomatic for realistic (large) datasets, versus Option A's single native pass.
- `details = {"null_counts": {col: count, ...}}`, filtered to only columns with `count > 0` (mirrors `InvalidDtypes`'s `mismatched_columns` filtering pattern — don't pad output with zeros).
- Severity is a single uniform `WARNING` for the whole rule, regardless of which column has nulls — **deferred design question**: a null `timestamp` is arguably more severe (unlocatable in time, breaks temporal rules) than a null `volume`/price (row still temporally locatable, could be forward-filled). Per-column/per-finding severity would be a first-of-its-kind pattern in this codebase (`ValidationRule.default_severity` is currently a single class-level attribute, one severity per rule); introducing it would require either multiple `ValidationResult`s per rule (not supported by `run_validation`, which expects exactly one result per `validate()` call) or computing an "effective severity" that overrides the class default per invocation. Left as `WARNING` uniformly for now; if this needs revisiting, the cleanest path is likely a **separate**, more specific rule (e.g. `structural.null_timestamps`) rather than retrofitting variable severity into this one.
- Message format leads with columns, not rows (consistent with `MissingColumns`/`InvalidDtypes`, which both lead with "the structurally wrong thing" first): `f"{len(null_counts)} column(s) contain null values, affecting {affected_rows} row(s): {null_counts}."`
- **Known limitation (not fixed, documented via test + this note)**: `coerce_dtypes()` casts numeric columns with `.cast(dtype, strict=False)`, so a value that fails to parse (e.g. a non-numeric string in a numeric column) is silently converted to `null` during ingestion. By the time `NullValues` runs, a genuinely-absent source value and a genuinely-corrupt-but-unparseable source value are indistinguishable — both just look like `null`. Fixing this is out of scope for this rule (it would mean changing `coerce_dtypes`, which affects every rule downstream, not just this one). Test `test_warn_unparseable_source_value_reported_as_null` documents this behavior explicitly so it's intentional, not an accidental blind spot.
- **Cross-rule interaction worth being aware of (not tested here, flagged only)**: an entirely-null column's effect on `InvalidDtypes`'s dtype check wasn't verified — if a fully-null column caused Polars to report an ambiguous/`Null` dtype instead of the coerced type, that would be an `InvalidDtypes` gap, not a `NullValues` bug. Worth checking if `InvalidDtypes` is revisited.

### Temporal (`validators/temporal.py`) — 3 of 4 implemented
| rule_id | Class | Status |
|---|---|---|
| `temporal.outside_trading_hours` | `OutsideTradingHours` | ✅ Implemented + fully tested |
| `temporal.missing_sessions` | `MissingSessions` | ✅ Implemented + fully tested |
| `temporal.gaps_within_session` | `GapsWithinSession` | ✅ Implemented + fully tested |
| `temporal.timezone_inconsistency` | `TimezoneInconsistency` | ⬜ Stub — **deliberately deferred**, see below |

**Infrastructure added with this rule — `MarketCalendar` now threaded via `RuleContext`.**
`RuleContext` (`validators/base.py`) gained a `calendar: MarketCalendar` field, defaulted
via `field(default_factory=MarketCalendar)` — the same pattern already used for
`config: ValidationConfig`. This fulfills `RuleContext`'s original docstring promise
("will be extended with calendar info… as rules are implemented") — it was the intended
extension point being exercised for the first time, not a new architectural decision.
- **Why context-injected rather than each rule constructing its own** (Option B over
  Option A): `engine/runner.py` already builds exactly **one** `RuleContext` per
  `run_validation()` call and shares it across every rule in the loop, so the calendar is
  constructed once per validation run instead of once per calendar-dependent rule. Since
  `MissingSessions` and `GapsWithinSession` will both need calendar access too, deciding
  this once here avoids re-litigating it three more times.
- Construction cost was measured before committing to `default_factory` (which fires on
  *every* `RuleContext()`, including in every test): `mcal.get_calendar("NYSE")` runs in
  **~0.03ms with no network/disk I/O** — it builds an in-memory calendar from bundled
  holiday rules. Safe to default-construct freely; confirmed no measurable test-suite
  slowdown.
- `engine/runner.py` needed **no change** — its existing `RuleContext(config=config)` call
  picks up the calendar via the default factory automatically.

`OutsideTradingHours` design decisions (for reference if extending/revisiting):
- **Scope is time-of-day ONLY, deliberately.** Whether the *date* is a valid trading
  session (weekend/holiday) is explicitly **not** checked here — that's `MissingSessions`'
  territory (in reverse: unexpected/extra sessions vs. missing ones). A Saturday timestamp
  at 10:00 therefore PASSES this rule. This is the same division-of-labor discipline as
  `UnsortedTimestamps`/`DuplicateTimestamps` splitting ownership of ties, and prevents two
  rules from double-reporting the same underlying problem. Locked in by
  `test_pass_weekend_date_not_flagged_time_of_day_only`.
- **Boundary is half-open `[09:30:00, 16:00:00)`**: exactly `09:30:00` IS regular hours
  (market opens then); exactly `16:00:00` is NOT (market is closed at that instant, last
  regular trade lands just before). Matches how exchanges actually define session bounds.
  Both boundaries have dedicated tests
  (`test_pass_exact_open_boundary_is_regular_hours`,
  `test_warn_exact_close_boundary_is_outside_hours`) so the inclusivity can't silently
  drift.
- **Timezone handling — the assumption being relied on, stated explicitly**: `dataset.df`'s
  `timestamp` column is tz-*naive* (`coerce_dtypes()` casts to `pl.Datetime("us")` with no
  tz), and `MarketCalendar.regular_open()`/`regular_close()` return plain tz-less `time`
  objects. So this is a **naive-to-naive comparison with no tz conversion at all**. That
  works because this pipeline assumes naive timestamps already represent **ET wall-clock
  time** — the data provider's contract, which this rule *uses* but does not *verify*.
  (Verifying it is exactly what `TimezoneInconsistency` would do, and exactly why that
  rule is currently blocked — see its deferral note below.) If that assumption is ever
  revisited, this rule's comparison logic must be revisited with it.
- Guard clauses: missing `timestamp` column → PASS/skip (`MissingColumns` owns absence);
  `df.height == 0` → PASS. Note this uses `== 0`, **not** the `< 2` guard used by
  `UnsortedTimestamps`/`DuplicateTimestamps` — those do pairwise/cross-row reasoning that's
  meaningless below 2 rows, whereas this rule evaluates each row independently, so a
  single row is perfectly checkable.
- Null timestamps are excluded **for free**, with no explicit filtering: `.dt.time()` on a
  null yields null, and Polars comparisons against null evaluate to null (not `True`), so
  nulls never enter the violation set. Same mechanism `UnsortedTimestamps` relies on for
  its shift comparison. `NullValues` still owns reporting nulls themselves.
- Implemented as a single vectorized Polars pass (`pl.col("timestamp").dt.time()` +
  filter on the two boundary comparisons), consistent with every other implemented rule —
  no Python row loops.
- **`INFO` severity → `WARN` status is a new pairing in this codebase** (previously only
  `CRITICAL`→`FAIL` and `WARNING`→`WARN` existed). Rationale: `default_severity = INFO`
  was already set in the stub, signaling this is informational — pre/post-market data
  isn't *invalid*, it's just outside the tool's stated "regular trading hours only" v1
  scope. `WARN` is the only non-PASS status milder than `FAIL`, so `INFO`→`WARN` extends
  the existing convention rather than inventing a third status tier.
- `details = {"violations": [{"index", "timestamp", "time_of_day"}, ...]}`, capped at
  `config.max_rows_in_details`; `affected_rows` holds the full uncapped count — mirrors
  `DuplicateTimestamps`'s shape (index + value + one descriptive field).
- Message format: `f"{affected_rows} row(s) outside regular trading hours
  ({regular_open}-{regular_close} ET) (first at index {i}: {time_of_day})."` — leads with
  the count, names the expected range so the reader knows what was compared against, then
  gives a concrete first example.

`MissingSessions` design decisions (for reference if extending/revisiting):
- Mechanic: `context.calendar.valid_sessions(start, end)` minus the set of dates actually
  present in `timestamp`. One schedule build for the whole range, **not** per-date
  `is_trading_day()` calls in a loop.
- **Range is derived from the dataset's own min/max non-null timestamp dates** (Option A),
  not from a user-declared expected range. Self-calibrating: a one-week file is never
  blamed for "missing" the rest of the year.
- **Known limitation (deliberate, documented — the tradeoff of Option A)**: because the
  range comes *from* the data, **edge truncation is undetectable**. A file intended to
  cover all January but stopping on the 20th can't have Jan 22–31 flagged, since those
  fall outside the derived range and the rule has no access to user intent. Options
  considered: **Option B** = user-declared `expected_start`/`expected_end` in
  `ValidationConfig`, **rejected for now** because `--config` is currently a confirmed
  no-op (see the defects list near the end of this doc), so the setting would be literally
  unreachable — building it would mean stacking a feature on a known-broken path.
  **Decision: Option C** = ship A now, document the gap, add B once `--config` actually
  loads. Same spirit as `NullValues` documenting its `coerce_dtypes` limitation rather
  than hiding it.
- **Scope: date-level presence only.** A session counts as present if it has ≥1 row. This
  rule never judges *coverage completeness within* a session — so **half sessions need no
  special handling**: the day after Thanksgiving (13:00 close) and Christmas Eve are just
  dates that are present or absent like any other. Partial intraday coverage is
  `GapsWithinSession`'s concern. Verified against the real NYSE calendar by
  `test_pass_half_session_counts_as_present` (2024-11-29 is a genuine half session).
- **Does NOT flag the inverse.** Dates *present* in the data that aren't valid sessions
  (weekend/holiday rows) are not reported — this rule is about absence, per its name.
  ⚠️ **That inverse case currently has no owner.** `OutsideTradingHours` explicitly
  disclaims date validity (it owns time-of-day only) and this rule disclaims extra dates,
  so "data on a day the exchange was closed" falls through the cracks. Candidate for a
  future rule (e.g. `temporal.unexpected_sessions`) following the same
  separate-rule-over-retrofit precedent that produced rule 14.
- Complements `OutsideTradingHours` cleanly: that rule owns **time-of-day**, this one owns
  **date presence**. The split was set up deliberately when `OutsideTradingHours` was
  written so neither double-reports.
- Null timestamps are dropped before deriving dates, so a null can neither collapse the
  range nor masquerade as a covered session (`test_warn_nulls_do_not_collapse_range`).
  `NullValues` owns reporting the nulls.
- **`affected_rows` is 0 even on violation** — missing sessions by definition contribute
  no rows to this dataset, so there are no rows to point at. Direct precedent:
  `MissingColumns` also uses `affected_rows=0` for an absent column. The count lives in
  `details["missing_count"]` and the message instead. Locked in by
  `test_warn_affected_rows_is_zero`.
- `details` carries more than just the list: `missing_sessions` (capped at
  `max_rows_in_details`), `missing_count` (uncapped total), plus
  `expected_session_count`, `present_session_count`, `range_start`, `range_end` — so a
  reader can see *what range was diffed* and judge whether the derived range matches
  their intent, which partially mitigates the edge-truncation limitation above.
- Dates are serialized to ISO strings (`str(d)`) in `details` rather than left as
  `datetime.date` objects, so the JSON report needs no custom encoder.
- Message format: `f"{n} trading session(s) missing between {start} and {end} (first:
  {first})."` — count, the range actually diffed, then a concrete first example.
- Test dates are pinned to **real NYSE calendar facts**, verified via `valid_sessions()`
  before writing assertions: 2024-01-06/07 weekend, 2024-01-15 MLK holiday, 2024-11-28
  Thanksgiving, 2024-11-29 half session. Tests assert against the actual exchange
  calendar rather than assumptions about it.

`SuspiciousPriceJump` design decisions — README-relevant:
- **Purpose: large close-to-close moves with no benign explanation**, surfaced as review
  candidates. Inherits `VolumeAnomaly`'s core ambiguity — a biotech dropping 70% on a
  failed trial is real — so `WARNING`, never `CRITICAL`, with review-oriented wording
  enforced by a test that forbids "corrupt"/"invalid".
- **Paired with `CorporateActionDiscontinuity`, mutually exclusive.** Both detect the same
  signal and divide it by *cause*: a split-shaped ratio at a session boundary belongs to
  the financial rule (`INFO` — data is unadjusted, not corrupt), anything else belongs
  here (`WARNING`). The stub severities already encoded this intent. The justification is
  that **the remedies differ**: an unadjusted split means "get adjusted data or apply the
  factor"; a bad tick means "remove or correct this point".
- **Shared definition in `validators/signatures.py`** (new module) so both rules use one
  split-ratio set. Duplicating it would drift and reintroduce double-reporting. Named for
  **splits only**, deliberately: see the dividend limitation below.
- **KEY REFINEMENT — the split exemption is gated on session boundaries.** A split takes
  effect at the start of a session, never mid-session. So an exact halving *within* a
  session is a bad tick and stays with this rule. Without that gate there would be a blind
  spot at precisely the ratios corruption produces (halving/doubling from a units or
  decimal error). Verified end-to-end: an overnight 100→50 is ignored as a 2:1 split while
  an intraday 50→25 — the identical ratio — is flagged.
- **Metric: the ratio `close[i]/close[i-1]`**, reported as a percentage change. Ratio
  because it is scale-free (a $5 move is meaningless without knowing if the stock is $10
  or $1000) and because split matching is naturally expressed in ratio space. Rejected
  absolute differences (not scale-free) and log returns (more symmetric but far less
  readable in a report, and the symmetry buys nothing for a threshold test).
- **Two thresholds selected by date change** — intraday 20%, overnight 50% (both
  configurable). Overnight moves are structurally larger, so one threshold would either
  flood at every session boundary or miss everything intraday. 20% intraday is justified
  by circuit breakers halting on 5–10% moves over five minutes; 50% overnight still
  catches order-of-magnitude errors (a decimal shift is +900%).
- **Frequency-agnostic by construction**: for daily data every consecutive pair *is* a
  date change, so the overnight threshold applies throughout automatically — same property
  that let `VolumeAnomaly` work without `inferred_frequency`.
- Rows with null close are excluded (`NullValues` owns nulls); rows where either close is
  `<= 0` are skipped since the ratio would be infinite or meaningless
  (`ImpossibleValues` owns those). Mirrors `VolumeAnomaly`'s zero-median guard.
- No `timestamp` column → boundaries are unknowable, so every transition is treated as
  overnight (looser) rather than manufacturing intraday violations.
- **Known gap (documented, unowned): single-bar "wick" spikes.** A bar with `high=10000`
  while open/low/close are ~100 is geometrically valid (so `OhlcRangeViolation` passes it)
  and does not move close-to-close returns (so this rule passes it too) — nothing detects
  it today. Left as a candidate future rule rather than overloading this one, per the
  precedent that produced rule 14.
- **Dividends are undetectable, by design and by name.** A dividend drop is 0.1–3% of
  price, indistinguishable from ordinary movement; no ratio test separates "fell 1.5% on a
  dividend" from "fell 1.5%". So `CorporateActionDiscontinuity` will detect **splits
  only** despite its name, and the shared helpers are named for splits so callers cannot
  mistake their coverage.

**Notes for `CorporateActionDiscontinuity` (next up):**
- It is the other half of the split above, so most decisions are already made: import
  `matches_split_ratio_expr`/`label_split_ratio` from `validators/signatures.py`, detect
  moves exceeding a threshold **at session boundaries only**, and report those whose ratio
  **matches** a split signature (the exact complement of what `SuspiciousPriceJump`
  reports).
- `INFO` severity → status should be `WARN` on finding, following the `INFO`→`WARN`
  precedent set by `OutsideTradingHours`.
- Use `label_split_ratio()` to name the inferred action (e.g. "2:1 split") in `details`,
  which is the rule's main value-add over the jump rule.
- Volume corroboration (a split often coincides with a volume shift) was considered but is
  NOT required by the current design — decide explicitly whether to add it as evidence.

`GapsWithinSession` design decisions — README-relevant:
- **Purpose: bars missing INSIDE a session** — 1-minute data that jumps 10:15 → 10:23.
  Distinct from `MissingSessions`, which owns whole absent days. Matters because a
  backtest that assumes every bar exists will silently skip the hole.
- **The prerequisite was solved by inferring the grid from the data itself**, not by
  populating `CanonicalDataset.inferred_frequency`. Inference lives in a new shared module
  `validators/frequency.py` (DD1 = option C), following the `signatures.py` precedent. Two
  reasons ingestion was rejected: (a) adding a rule should not require touching ingestion,
  and (b) `inferred_frequency` is a **`str`** while gap arithmetic needs a `timedelta`, so
  populating it would be a lossy round-trip requiring a format parser nothing else needs.
  The field remains unpopulated; ingestion may later call `format_frequency()` for display.
- **Frequency = the MODE of within-session deltas.** Rejected `min` (one odd row sets the
  grid), `mean` (skewed by the very gaps being hunted), `median` (fails past 50% loss). The
  mode is right because the intended spacing is by definition the most frequent one.
- **Delta-based, not grid-based** (DD3). Comparing consecutive bars avoids depending on
  vendor conventions — whether a 16:00 bar exists, whether bars are stamped at interval open
  or close, half-session closes — any of which would generate false positives on good data.
  Consequence documented, not hidden: **truncation at session edges is not detected** (a
  session starting late or ending early). Same shape of limitation already accepted for
  `MissingSessions`.
- **No calendar dependency at all**, which is precisely what makes half sessions a
  non-issue: a 13:00 close simply yields fewer bars and leaves every within-session delta
  untouched. Same self-solving property `MissingSessions` had.
- **Daily data self-skips with zero special-casing.** Every delta spans a session boundary,
  so no within-session interval survives the filter, inference fails, and the rule skips —
  correct, because "gap within a session" is meaningless when a session holds one bar.
- **Dominance guard is the most important safeguard** (DD4). The modal spacing must be a
  **strict majority** of intervals or the rule skips. Without it, tick/event data (no fixed
  grid, arbitrary modal delta) would have nearly every interval reported as a gap. The
  comparison is strict so a perfect two-way tie is rejected as ambiguous rather than
  resolved arbitrarily by the tie-break.
  - **Caught during implementation:** the first version used `confidence < dominance`, which
    *accepted* a 50/50 tie (0.5 is not < 0.5) and returned a frequency on ambiguous data.
    Corrected to `<=`; `test_pass_ambiguous_tie_skipped` pins it.
- **Determinism defect avoided:** polars `Series.mode()` can return several values in
  **non-deterministic order** on a tie, which would make the rule answer differently on
  identical input. The helper uses `value_counts()` with an explicit sort (count desc, then
  smallest delta) instead. Pinned by `test_tie_break_is_deterministic`.
- **Sorted, de-duplicated copy** (DD5). Rules run independently and cannot assume sorted
  input — the reasoning that drove `DuplicateTimestamps` to group-by. Duplicates are dropped
  because a zero delta is not a gap and would corrupt the modal spacing. Consequence:
  reported `index` is **sorted** position, so timestamps are the primary locator.
- **`missing_bars = delta/frequency - 1`.** The `-1` is the crux: a 4-minute delta on a
  1-minute grid hides **3** bars, not 4, because `gap_end` itself is present. An off-by-one
  here would look entirely plausible in a report, so
  `test_warn_missing_bars_arithmetic_is_not_off_by_one` guards it.
- **`largest_gap_bars` is the key diagnostic field.** It separates 47 scattered single-bar
  holes (benign illiquidity — vendors often omit zero-volume minutes) from one 47-bar hole
  (an outage or truncated feed). Identical `total_missing_bars`, completely different
  implications.
- **`frequency_confidence` makes the inference auditable** — reported on PASS, on WARN, and
  even when inference is *rejected*, so a skip is explainable. Parallels
  `CorporateActionDiscontinuity` echoing `split_ratio_tolerance`. If it reports `"5min"` on
  data you believe is 1-minute, you have learned something important.
- **Durations are pre-formatted strings**, not `timedelta`. Verified: a raw `timedelta` in
  `details` serialises to ISO-8601 `"PT4M"` under `model_dump(mode="json")`, and plain
  `json.dumps` rejects it outright — `render_json` only survives via `default=str`.
- `affected_rows = gap_count` (existing rows following a gap), never total missing bars,
  which would break the distinct-rows convention and can exceed `row_count`. Pinned by
  `test_warn_affected_rows_never_exceeds_row_count`.
- WARNING, worded for review: a genuine trading halt produces a legitimate gap and this rule
  cannot distinguish one from a truncated feed.
- **Second honest limitation, tested:** mode inference assumes gaps are the minority. When
  they are not — e.g. bars at every *second* minute — the gap spacing itself becomes the
  inferred grid and the rule under-reports rather than guessing. Documented by
  `test_pass_documents_limit_when_gaps_outnumber_real_intervals`.

New config fields: `gap_min_missing_bars=1` (report every gap; single missing bars are
common but still affect backtests), `gap_frequency_dominance=0.5` (strict majority).

## Reporting-integrity work (defects 1-5, 7, 8, 10) — DONE

The tool's own reporting was fixed before finishing the rule set, on the reasoning that a
fourteenth check adds little to a tool that cannot be trusted to report the result.

**`Status.SKIP` added — the central change.** Previously two distinct populations were both
reported as `PASS`: unimplemented rules (1) and rules that ran but could not evaluate (11
sites across 8 rules). Both were the same lie — "I did not verify this" reported as "this is
fine". The decisive argument for fixing both: population (1) is temporary and vanishes once
`TimezoneInconsistency` ships, while the 11 guard sites are permanent. Later extended to the
height guards (10 more sites), so a rule that cannot form a pair or has no rows now skips.
  - `SKIP` never worsens the overall status (a rule that could not run found no problem), but
    is never counted as a pass either.
  - Kept as PASS deliberately: `DuplicateTimestamps` when `dup_counts` is empty, and
    `NullValues` on a zero-row frame with all columns present — those are real verdicts.

**Exit codes: 0 clean / 1 validation failed / 2 tool error.** Previously the CLI exited 0
even on `FAIL`, making it silently useless in CI, while file-load errors used 1 — colliding
with the meaning "data is bad". Tool errors moved to 2, matching Typer's own usage-error code.

**`--strict` implemented** (was a silent no-op) — `WARN` exits 1 under `--strict`, 0 otherwise.
Warnings do not fail by default because they cover findings needing human judgement: a genuine
halt looks like a gap, a real biotech collapse looks like a bad tick.

**`--config` implemented** (was a silent no-op). Flat TOML via stdlib `tomllib` — chosen over
JSON because a tuning file needs comments, and over YAML to avoid a dependency. CLI flags take
precedence so a one-off `--strict` overrides a committed file. **Unknown keys are rejected**
(`extra="forbid"`): pydantic's default silently drops them, so a typo like
`volume_anomly_multiplier` would leave the user believing they had configured something they
had not — the same quiet dishonesty as reporting an unrun check as a pass. `output_format`,
`output_path`, and `strict` are rejected from the file: they describe how the tool was invoked,
so a committed config must not redirect another user's output.

**`severity_overrides` deleted rather than implemented.** Each rule declares one class-level
severity AND hardcodes its status, so overriding severity alone yields incoherent results
(`severity=critical, status=warn`), and deriving status from severity would destroy the
deliberate `INFO`→`WARN` pairings. The needs it implied are served by `--strict` and
`disabled_rules`.

**Zero-row and nothing-evaluated outcomes.** A header-only CSV used to report `pass` with 13
passes. Some rules legitimately reach verdicts on an empty frame (the columns exist, so the
schema is genuinely checkable), so a per-rule fix was insufficient — the aggregator now treats
`row_count == 0` as `WARN`. New `DatasetSummary.overall_message` explains outcomes that no
individual rule accounts for, rendered as a `Note:` line.

**Disabled rules are reported as skips, not omitted.** Otherwise a reader cannot tell "14
rules, 3 disabled" from "this tool has 11 rules". Category filters still omit silently, since
selecting one category is an explicit narrowing rather than a hidden gap.

### `MissingColumns` was unreachable through the CLI (found while implementing, worst of the lot)

`coerce_dtypes()` raised `ValueError` on any missing required column, **before validation ran**.
Consequences: the `MissingColumns` rule could never fire via the CLI; the
"MissingColumns owns absent columns" guards in 8 rules were likewise unreachable; the user got
a raw traceback instead of a report; and the exit code was 1, wrongly meaning "validation
failed". Fixed by making `coerce_dtypes()` tolerant — reporting missing columns is a validation
concern, not an ingestion concern — with a CLI backstop that still reports genuinely unreadable
files as tool errors (exit 2). A column-missing file now produces a real report naming exactly
what is absent, with dependent rules skipping.

### Toolchain
`ruff check .` passes for the first time (4 pre-existing errors fixed). **mypy installed and
`mypy src` is clean under `strict = true`** (11 real errors fixed): `EXPECTED_DTYPES` annotated
with polars' own `PolarsDataType` union since it mixes dtype classes with parameterised
instances; `Series.min()/max()` narrowed to `datetime`; bare `dict` annotations parameterised.
Two config bugs fixed: `python_version` was pinned to 3.11 while the venv runs 3.14 (which made
mypy reject numpy's own stubs), and `pandas_market_calendars` needed an ignore-missing-imports
override.

### Test suite: 315 → 388
Plumbing was the least-tested code in the project and where every defect lived.
- `test_engine.py`: 3 → 26. The old `test_run_validation_with_stubs` passed for the wrong
  reason — it asserted "all rules are stubbed" and only held because the fixture was clean.
  Now covers `disabled_rules`, `enabled_categories`, the skip path, the generic-exception path
  (via a temporarily-registered broken rule), counter reconciliation, status precedence, and
  the zero-row and all-skipped outcomes.
- `test_cli_end_to_end.py`: 6 → 20. The old `--strict` test asserted only that the flag "did
  not crash", which is precisely why the no-op survived; it now asserts behaviour. Adds the
  full exit-code matrix and config-file integration.
- `test_reporting.py`: **new, 25 tests.** The reporting layer previously had none.
- `test_config_file.py`: **new, 14 tests.**

### `InvalidDtypes` reported a vacuous pass (found after the SKIP sweep)

`structural.invalid_dtypes` returned `PASS` with "All column dtypes match the expected
schema" **after inspecting zero columns**. Its mismatch comprehension filters on
`if col in df.columns`, so a file with no expected columns produced an empty mismatch set,
which fell through to the PASS branch. A file of `alpha,beta,gamma` therefore reported
`missing_columns: fail` alongside `invalid_dtypes: pass`.

Missed in the original sweep because both greps keyed on markers this rule lacks — a
`"skipped."` message and a `height` guard. It was the only rule in the registry with no skip
path other than `missing_columns` (correct by design: it inspects the column list, which
always exists) and the stub.

Two fixes:
1. **Skip when nothing is inspectable** — no expected columns present now yields `SKIP`,
   mirroring the `NullValues` `if not checked_cols` guard.
2. **Pass message states coverage** — "All 6 expected columns have correct dtypes" versus
   "All 1 of 6 expected column(s) present have correct dtypes". The old wording overstated a
   check of one column as verification of the whole schema. Status is unchanged: the rule did
   genuinely verify what was there.

Verified not to be bugs while investigating: `InvalidDtypes` does **not** double-report absent
columns (it ignores them, so `MissingColumns` retains ownership), and unparseable text such as
`close = "abc"` casting to `Float64`-with-nulls is the documented limitation already covered by
`test_warn_unparseable_source_value_reported_as_null`, not a regression from making
`coerce_dtypes()` tolerant.

**`TimezoneInconsistency` — deliberately deferred, not just "not yet gotten to."**
Investigated first (before `OutsideTradingHours`) since it looked like the
lowest-dependency temporal rule (no `MarketCalendar` needed). That
investigation surfaced a real blocker worth recording so it isn't
re-discovered from scratch later:

- `ingestion/schema.py`'s `coerce_dtypes()` casts the `timestamp` column to
  `pl.Datetime("us")` — **no timezone** — via `.cast(dtype, strict=False)`,
  unconditionally, for every dataset. Polars columns have exactly one dtype;
  a single `Series` can never hold a per-row mix of naive/aware timestamps.
  So whatever timezone information existed in the source file (mixed
  offsets, mixed naive/aware rows, etc.) is **silently discarded during
  ingestion**, before any validator — including this one — ever sees the
  data. By the time `TimezoneInconsistency.validate()` would run, there is
  nothing left to detect: the inconsistency (if any existed) already
  happened upstream and left no trace in `CanonicalDataset`.
- Two options were identified:
  - **Option A (real fix, deferred)**: change ingestion to preserve
    pre-coercion timestamp representation (e.g. load `timestamp` as raw
    strings instead of using `load_csv`'s `try_parse_dates=True`, thread a
    raw/pre-cast column or parse-diagnostics side-channel through
    `CanonicalDataset`) so the rule can detect genuine mixed naive/aware or
    mixed-offset rows. This is the only option that makes "inconsistency"
    mean what the rule name implies. **Explicitly out of scope for now** —
    it touches `ingestion/loaders.py`, `ingestion/schema.py`,
    `models/dataset.py`, and `ingestion/canonicalize.py`, i.e. exactly the
    "ingestion, ~~not just validators~~" blast radius this codebase's
    architecture is designed to avoid for a single rule, and it changes
    pipeline behavior for every dataset, not just this rule's concern.
  - **Option C (rejected for now)**: keep ingestion as-is and scope the rule
    down to column-level tz metadata (naive-vs-aware dtype check, which is
    always trivially true given current coercion and therefore nearly
    useless) or a wall-clock-plausibility heuristic (which conceptually
    overlaps with `OutsideTradingHours`'s territory — both would be
    reasoning about "does this wall-clock hour make sense," a division of
    labor that would need to be defined explicitly, the same way
    `UnsortedTimestamps`/`DuplicateTimestamps` divided ownership of ties).
- **Decision: defer this rule entirely** rather than ship Option C's
  narrowed/overlapping version. `OutsideTradingHours` was implemented
  instead (its design notes are above). Note that Option C's overlap concern
  is now concrete rather than hypothetical: `OutsideTradingHours` has since
  claimed time-of-day plausibility as its own scope, so a future
  `TimezoneInconsistency` must **not** re-derive "is this wall-clock hour
  sensible" — it needs to detect genuine source-level tz representation
  disagreement (Option A), which is a different question and doesn't
  collide. Revisit when ready to take on Option A's ingestion change as its
  own deliberate piece of work — read this note first before restarting that
  design, so the blocker isn't rediscovered from scratch.
- Worth noting for whoever picks up Option A: `OutsideTradingHours` now
  *depends* on the "naive timestamps are ET wall-clock" assumption (see its
  notes above). Option A would make that assumption verifiable rather than
  merely assumed, so the two rules are complementary — but if Option A ever
  changes the canonical `timestamp` dtype to be tz-*aware*,
  `OutsideTradingHours`'s naive-to-naive comparison against
  `regular_open()`/`regular_close()` would need updating in lockstep.

`temporal.missing_sessions` (`MissingSessions`) and `temporal.gaps_within_session`
(`GapsWithinSession`) remain stubs, not yet investigated in detail.

### Numerical (`validators/numerical.py`) — 4 of 4 implemented ✅
| rule_id | Class | Status |
|---|---|---|
| `numerical.ohlc_range_violation` | `OhlcRangeViolation` | ✅ Implemented + fully tested |
| `numerical.volume_anomaly` | `VolumeAnomaly` | ✅ Implemented + fully tested |
| `numerical.suspicious_price_jump` | `SuspiciousPriceJump` | ✅ Implemented + fully tested |
| `numerical.impossible_values` | `ImpossibleValues` | ✅ Implemented + fully tested (14th rule) |

`VolumeAnomaly` design decisions (light notes — README-relevant):
- **Core purpose: flags review candidates, not defects.** It cannot distinguish a
  data error from a genuine market event — a 50x spike is earnings day *and* a
  units bug. Hence `WARNING` (never `CRITICAL`), and the message is worded
  "may warrant review", asserted by a test that forbids the words
  corrupt/invalid/error. A rule that cries corruption on every earnings day
  destroys trust in the whole report.
- **Method: `volume > k × centered rolling_median(volume, window)`.** Rejected
  alternatives, with reasons worth keeping: global **z-score is actively broken**
  here (volume is right-skewed, and one 1000x outlier inflates `std` enough to
  hide itself — it fails hardest on exactly what you want to catch); global
  median/MAD/IQR fix skew but miss the decisive issue.
- **The decisive issue — intraday volume is U-shaped** (open ~30x midday, close
  ~15x), so any *global* threshold flags every open/close bar. Measured on a
  realistic 3-session, 1,170-bar minute series: rolling → **0** false positives,
  global 10x-median → **132**, global z-score>3 → **45**. Locked in by
  `test_pass_realistic_intraday_u_shape_no_false_positives`, which fails if
  anyone swaps in a global statistic.
- Rolling also sidesteps the fact that the rule **cannot know the bar frequency**
  (`CanonicalDataset.inferred_frequency` is never populated), since a local
  window is sensible at any frequency. Median (not mean) prevents self-masking;
  multiplicative (not additive) keeps it scale-free across liquidity levels.
- **Defaults `k=10`, `window=20`, deliberately calibrated**: genuine events run
  2–5x, targeted data errors run 100x+, so 10x sits in the gap. Both configurable
  via `ValidationConfig` (`volume_anomaly_multiplier`, `volume_anomaly_window`) —
  not settable from the CLI yet since `--config` is a no-op, same caveat as
  `MissingSessions`.
- **Zero-median guard = the trap in this rule.** In an illiquid/halted stretch the
  rolling median is 0, and `v > k*0` would flag every positive bar — a silent
  false-positive explosion. Rows with median <= 0 are skipped; dedicated test.
- **Centered window** (look-ahead bias is irrelevant when auditing a finished
  file), **high-side only in v1** (low-side roughly doubles false positives on
  thin midday bars; documented as a possible extension, with a test asserting a
  1-share collapse is *not* flagged), and **partial windows at the edges**
  (`min_samples=1`) so the first/last bars aren't permanent blind spots.
- `details` echoes back `multiplier` and `window` alongside per-violation
  `volume`/`rolling_median`/`ratio`, so a reader can see what threshold produced
  the finding.

`ImpossibleValues` (rule 14) design decisions — README-relevant:
- **Purpose: unambiguously impossible values**, as distinct from merely unusual ones.
  Findings need no hedging (unlike `VolumeAnomaly`): a negative price is simply wrong,
  hence `CRITICAL` → `FAIL`.
- **Rescoped from `non_positive_prices`** while still a stub (so it cost nothing) to also
  own `volume < 0`. Reason: negative volume is unambiguous and deserves `CRITICAL`, but
  `VolumeAnomaly` is statistical and carries `WARNING` — and **one rule carries exactly
  one severity** (`default_severity` is a single class attribute), so they cannot share a
  rule without mis-reporting one. Grouping impossible prices with impossible volume *is*
  right: same severity, same remedy (go back to the source).
- **Deliberate asymmetry — prices at `<= 0`, volume only at `< 0`.** `volume == 0` is
  legitimate (illiquid names, halted trading, a session with no prints), so flagging it
  would create false positives on good data. A price of exactly 0 is never a real trade.
- **NaN detection — closes a verified hole and fixes a misdiagnosis.** Polars uses a
  **total order in which NaN sorts above all numbers — NOT IEEE-754** (`NaN > 1.0` is
  True, `NaN == NaN` is True). Measured consequences: `NullValues` passes NaN in every
  column (`is_null()` is False for NaN); `InvalidDtypes` passes it (dtype is still
  Float64); `NaN <= 0` is False so a plain non-positive predicate misses it; and
  **`OhlcRangeViolation` passes a NaN in `high` entirely** (NaN sorts above everything, so
  no `high < ...` check fires) while *misdiagnosing* NaN in open/low/close as a geometry
  error like `high < close`. So without an explicit `is_nan()` check, NaN in `high` is
  undetectable by any rule here. Both behaviours are pinned by tests.
  A CSV containing the literal text `NaN` reaches validation as a real NaN via
  `coerce_dtypes`, so this is a live ingestion path. NaN matters more than a missing value
  because it is *contagious* — it propagates through every downstream average, return and
  indicator, so a backtest yields NaN metrics or silently drops rows instead of failing.
  Applies to the four price columns only; `volume` is Int64 and cannot hold NaN.
- **No-trade bars are tagged, not hidden (Option B).** Some vendors encode a no-trade bar
  as `open=high=low=close=0` with `volume=0`. Those rows are still reported — they are
  genuinely unusable (a backtest computes a −100% return, then divides by zero) — but
  labelled `"no-trade bar (all prices zero)"` instead of emitting four redundant
  `"<col> <= 0"` findings. Same detection and severity, far more actionable report: a
  reader can tell at a glance whether they have 5,000 vendor no-trade bars or 3 corrupt
  prices. Option C (exempt them) was rejected on principle — it would return a clean
  report on unusable data, the same failure this rule exists to close.
- The pattern requires the **full OHLC set plus volume, with `volume == 0`**, so all-zero
  prices *with* positive volume (contradictory, genuinely corrupt) are still reported as
  individual violations, and a partial-column dataset can't be mislabelled.
- `details["no_trade_bar_count"]` is computed over the **full** violation set, not the
  capped sample, so the summary stays accurate when details are truncated.
- Otherwise mirrors `OhlcRangeViolation`: vectorized `any_horizontal`, distinct-row
  `affected_rows`, `details` capped at `max_rows_in_details`, Python work only on the
  capped subset, check names encoding the violating condition (`"open <= 0"`,
  `"high is NaN"`, `"volume < 0"`) so the asymmetry is visible in the report itself.

**Rule count is now 14, not 13.** `numerical.non_positive_prices`
(`NonPositivePrices`) was added beyond the scaffold spec's original 13 as a
deliberate scope decision while implementing `OhlcRangeViolation` — see that
rule's notes below for why it's separate rather than folded in. This is the
same "separate rule rather than retrofit a second concern into an existing
one" reasoning this document already floated for a possible
`structural.null_timestamps`.

`OhlcRangeViolation` design decisions (for reference if extending/revisiting):
- **Invariants enforced** — three, expanded into five elementary checks:
  `high >= low`, `high >= max(open, close)`, `low <= min(open, close)`, expanded to
  `high < low`, `high < open`, `high < close`, `low > open`, `low > close`.
  Kept **expanded rather than collapsed** into `max()`/`min()` so the report can name
  exactly which relationship broke — `high < low` is a materially different pathology
  (usually a swapped-column bug) than `close > high` (usually a bad tick or an
  adjustment artifact), and a user fixing their data needs to know which.
- `high >= low` is checked **explicitly**, not left to transitivity through open/close:
  if `open`/`close` are null but `high`/`low` are present and inverted, the open/close
  comparisons evaluate to null (and are excluded), while `high < low` still catches the
  genuine problem. Dropping the explicit check would create a blind spot on exactly the
  rows that are already partially corrupt.
- **Scope: single-row relative geometry only.** No cross-row comparison (that's
  `SuspiciousPriceJump`'s territory) and no `volume` (`VolumeAnomaly` owns it).
- **Non-positive prices deliberately excluded → became rule 14.** A bar like
  `open=-5, high=-1, low=-10, close=-3` satisfies *every* geometry invariant while being
  economically impossible for an equity. Three options were weighed: (A) leave it out of
  scope entirely, (B) fold a `> 0` check into this rule, (C) split it into its own rule.
  **B was rejected** because it would make one `rule_id` report two unrelated failure
  kinds through one `details` shape, muddying the report — a sign error and a geometry
  error need different explanations and different fixes. **Decision: A now + C
  immediately**, i.e. this rule stays geometry-only and `numerical.non_positive_prices`
  was stubbed in the same commit so the gap has a named owner rather than silently
  belonging to nobody. Locked in by
  `test_pass_negative_prices_not_flagged_here`, which asserts a
  geometrically-valid negative bar PASSES here — so the boundary is intentional and
  visible, not an accidental blind spot.
- **Column-subset handling**: only checks whose *both* operand columns exist are built,
  mirroring `NullValues`' "check what's present" approach rather than skipping the entire
  rule when one column is absent. With only `high` and `low` present, `high < low` still
  runs (`test_fail_partial_columns_still_checked`). `MissingColumns` owns reporting
  absence.
- Guard: if no check has both operands present, or `df.height == 0` → PASS. Uses
  `== 0` rather than `< 2` since each row is evaluated independently — same reasoning as
  `OutsideTradingHours`.
- Nulls excluded for free (Polars comparisons against null yield null, not `True`), same
  mechanism `UnsortedTimestamps`/`OutsideTradingHours` rely on. `NullValues` owns nulls.
- **Equality is valid, not a violation** — all invariants use `>=`/`<=`, so a flat bar
  (`open == high == low == close`, common in illiquid names or halted trading) passes, as
  does `open == low` / `close == high`. Both covered by tests so this can't regress into
  false positives on legitimate data.
- `affected_rows` = count of **distinct rows** with ≥1 failed check, not the sum of failed
  checks across rows (a row failing 3 checks counts once) — mirrors `NullValues`'
  row-vs-column semantics. Locked in by
  `test_fail_row_with_multiple_violations_counted_once`.
- `details = {"violations": [{"index", <present OHLC values>, "failed_checks": [...]}, ...]}`,
  capped at `config.max_rows_in_details`. Carries the actual price values, not just
  indices, so a reader can diagnose without re-opening the source file. Python-side
  iteration happens **only on the capped subset**; detection and counting stay fully
  vectorized, so a million-row file with a million violations still only boxes
  `max_rows_in_details` rows into Python.
- `CRITICAL` → `FAIL` (severity was already set in the stub): a malformed bar isn't usable
  and can't be safely auto-repaired, consistent with the `CRITICAL`→`FAIL` pairing used by
  `MissingColumns`/`InvalidDtypes`/`DuplicateTimestamps`.
- Message format: `f"{affected_rows} row(s) have invalid OHLC ranges (first at index {i}:
  {', '.join(failed_checks)})."` — count first, then a concrete first example naming the
  broken relationship(s).

### Financial (`validators/financial.py`) — 1 of 1 implemented ✅
| rule_id | Class | Status |
|---|---|---|
| `financial.corporate_action_discontinuity` | `CorporateActionDiscontinuity` | ✅ Implemented + fully tested |

`CorporateActionDiscontinuity` design decisions — README-relevant:
- **What a finding means: the data looks RAW, not broken.** If a dataset were split-adjusted
  there would be no discontinuity to see, so the presence of one *is* the evidence of
  non-adjustment. Remedy: obtain adjusted data or apply the factor — as opposed to a bad
  tick, which you delete or correct. That difference in remedy is the whole reason this is
  a separate rule from `SuspiciousPriceJump`.
- **Exact complement of `SuspiciousPriceJump`.** Split-shaped ratio at a session boundary →
  this rule (`INFO`); anything else → the jump rule (`WARNING`). Mutual exclusion holds
  because the jump rule excludes *every* boundary split-shaped ratio regardless of
  magnitude (its exclusion is not threshold-gated). Guaranteed by a dedicated
  `TestGapRulePartition` suite asserting each scenario is owned by exactly one rule, and
  that a benign move is owned by neither.
- **NO magnitude threshold — the signature IS the filter.** Every split ratio is far from
  1.0 (closest is 4:3 at 0.75, whose ±2% band is 0.735–0.765), so ordinary daily returns
  (~0.95–1.05) cannot match. Specificity comes from ratio matching, making a threshold
  redundant.
- **This avoided a real gap.** Reusing the jump rule's 50% overnight threshold would have
  meant a 3:2 split (−33.33%) and a 4:3 split (−25%) were reported by **neither** rule —
  genuinely unadjusted data receiving a clean bill of health. Pinned by
  `test_small_split_is_not_missed_by_both`.
- **Boundary-only**: a split takes effect at the start of a session, never mid-session, so
  an intraday halving is a bad tick and stays with the jump rule.
- **Volume corroborates but never filters.** `volume_ratio` across the transition is
  reported as evidence (a 2:1 split roughly doubles share volume). Requiring it would cause
  false negatives whenever a vendor restates volume or the shift hides in normal variance —
  trading real detections for precision this rule doesn't need at `INFO`. Pinned by
  `test_warn_split_detected_even_when_volume_uncooperative`.
- **The inference is an inference.** With no corporate-actions feed (v1 forbids external
  data), findings say "consistent with a 2:1 split", never "a 2:1 split occurred". `details`
  exposes the raw `ratio` and the `split_ratio_tolerance` used so a reader can audit the
  reasoning instead of trusting the label. Message wording is asserted by a test that
  requires "unadjusted"/"consistent with" and forbids "corrupt"/"invalid"/"error".
- **Splits only — dividends are undetectable**, and the rule says so rather than implying
  coverage it lacks. A dividend drop is 0.1–3% of price, indistinguishable from ordinary
  movement; no ratio test separates "fell 1.5% on a dividend" from "fell 1.5%". Documented
  by `test_pass_dividend_sized_drop_not_detected`.
- `INFO` → `WARN` status, following the precedent set by `OutsideTradingHours` (no
  `Status.INFO` exists, and `FAIL` would overstate unadjusted-but-valid data).
- Requires `timestamp` to identify sessions; without it the rule skips rather than guessing.
  Null closes excluded (`NullValues`), non-positive closes skipped (`ImpossibleValues`).
- `inferred_action_counts` is tallied over the **full** candidate set, so summaries stay
  accurate when `details` truncates — same pattern as `ImpossibleValues`' no-trade count.

## Test status

`pytest` → **392 passed, 2 skipped, 0 failed** (last run confirmed this session). The
1 skip is `TestStructuralRulesRegistered::test_stub_rules_raise_not_implemented`,
which is parametrized over `STUB_STRUCTURAL_RULES` — now empty since all 5
structural rules are implemented, so pytest emits a harmless empty-parametrization
skip rather than an error.
`tests/unit/test_structural.py` has full PASS/FAIL/WARN/message-format coverage for
`MissingColumns`, `InvalidDtypes`, `UnsortedTimestamps`, `DuplicateTimestamps`, and
`NullValues` — all 5 structural rules. Use this file as the template for testing
style/conventions when implementing new rules.

`tests/unit/test_temporal.py` now follows the same structure: full PASS/WARN/
message-format coverage for `OutsideTradingHours` (15 dedicated tests, including
both half-open boundary cases and the weekend/time-of-day-scope test), plus
registration/category/`NotImplementedError` contract tests for the 3 remaining
temporal stubs via `STUB_TEMPORAL_RULES`.

`tests/unit/test_numerical.py` follows the same structure: full PASS/FAIL/
message-format coverage for `OhlcRangeViolation` (24 dedicated tests, including all
five elementary checks, equality-is-valid boundary cases, the deliberate
negative-price scope boundary, and partial-column handling), plus
registration/category/`NotImplementedError` contracts for the 3 numerical stubs.

`test_financial.py` still only asserts stub/registration contracts, not real logic.

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

**Agreed implementation timeline** (working through it in this order):

1. ~~`OhlcRangeViolation`~~ — ✅ **done** (geometry only; spun off rule 14 for signs)
2. ~~`MissingSessions`~~ — ✅ **done** (date-presence only; range self-derived)
3. ~~`VolumeAnomaly`~~ — ✅ **done** (rolling-median baseline; spun rule 14 rescope out of it)
4. ~~`SuspiciousPriceJump`~~ — ✅ **done** (rule 14 `ImpossibleValues` also done, slotted in here)
5. ~~`CorporateActionDiscontinuity`~~ — ✅ **done** (paired half of the gap-detection split)
6. ~~`GapsWithinSession`~~ — ✅ **done** (frequency inference solved via `validators/frequency.py`)
7. **`TimezoneInconsistency`** ← last rule — blocked on the ingestion tz change (Option A)
5. `CorporateActionDiscontinuity`
6. `GapsWithinSession`
7. `TimezoneInconsistency`

Plus `NonPositivePrices` (rule 14) — small, scope already decided, can slot in
anywhere; see its note in the Numerical section.

Rationale for the ordering: start with the only rule that had zero open design
questions, then take the moderate calendar rule while `context.calendar`
plumbing is fresh, then the two threshold-based numerical rules (deciding the
jump-vs-corporate-action boundary once, which covers both), then the financial
rule that builds on `SuspiciousPriceJump`. The two with genuine prerequisites —
bar-frequency inference and the ingestion tz change — go last.

**Notes for `SuspiciousPriceJump` (next up):**
- Needs a threshold decision, like `VolumeAnomaly` — reuse that rule's reasoning
  (robust local baseline, multiplicative/returns-based, configurable, guard the
  degenerate-baseline case).
- **The boundary with `CorporateActionDiscontinuity` must be settled first**: both
  detect the same *signal* (a large price gap) with different *causes*. A ~50% drop
  with ~2x volume is probably a 2:1 split, not bad data. Decide once, covering both.
- Likely operate on **returns** (close-to-close), not raw price differences, so it's
  scale-free — same reasoning that made `VolumeAnomaly` multiplicative.
- Overnight gaps between sessions are normal and much larger than intraday
  bar-to-bar moves; treating them identically would produce a false positive at
  every session boundary. This is the direct analogue of `VolumeAnomaly`'s U-shape
  problem and needs an explicit decision.

## Known defects in already-"complete" code (found during review, NOT yet fixed)

These are in shipped code, not stubs, and matter more than any remaining rule:

1. **Unimplemented stub rules are reported as PASSES.** `engine/runner.py` catches
   `NotImplementedError` and substitutes a `Status.PASS` placeholder, so a report
   claims e.g. `total_passed: 12` when only a handful of rules actually ran.
   `DatasetSummary.total_skipped` exists for exactly this and is hardcoded to `0`.
   For a tool whose entire value is trustworthy reporting, this is the most damaging
   bug present. Gets less visible with each rule implemented but never self-resolves.
2. **CLI exit code is always 0**, even when `overall_status == FAIL` — no
   `raise typer.Exit(code=1)` in `cli/main.py`. Makes the tool unusable in CI,
   pre-commit hooks, or any shell pipeline.
3. **`--strict` is a silent no-op.** Plumbed into `ValidationConfig.strict`; nothing
   reads it. Verified: a WARN dataset returns `warn` identically with and without it.
4. **`--config PATH` is a silent no-op.** Accepted as an option, never read;
   `validation_config` is built from the other flags only.
5. **`severity_overrides`** exists in `ValidationConfig`, applied nowhere.
6. **4 ruff errors** (2 auto-fixable): `E501` in `calendar/timezones.py:46` and
   `engine/runner.py:22`; `I001` import order in `ingestion/loaders.py` and
   `validators/structural.py`. README documents `ruff check .` as part of the
   workflow and the scaffold spec listed a clean pass as an acceptance criterion,
   so this is a broken claim rather than style preference.
7. **mypy is in `[dev]` deps but not installed** and appears never to have been run,
   despite the codebase being fully annotated.

## A note on this document's purpose going forward

**This file is the source material for refining the README at project culmination.**
The per-rule design notes are written with that in mind: the "core purpose",
"what it covers / deliberately does not cover", and calibration-rationale bullets
are intended to be distillable into user-facing documentation. Keep new notes in
that spirit — a reader should be able to derive *what the tool checks and why it
can be trusted* from this file alone.

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
pytest                                   # run tests (392 passing, 2 skipped baseline)
ruff check .                             # lint
marketcheck validate <file> --format json
```
