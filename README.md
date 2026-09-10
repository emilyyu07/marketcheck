# MarketCheck

A command-line tool that inspects historical US-equity OHLCV data and reports whether it is
structurally and financially trustworthy enough for research or backtesting.

**A check that did not run is never reported as a pass.** Reports distinguish "verified and fine"
from "could not be verified" — a tool you consult to decide whether to trust your data is worthless
if it overstates what it checked.

## Quick start

```bash
uv sync                                  # or: pip install -e ".[dev]"

marketcheck AAPL_1min.csv                # text report
marketcheck AAPL_1min.csv -f json        # machine-readable
marketcheck AAPL_1min.csv --strict       # warnings also fail the build
```

Accepts `.csv`, `.parquet`, and `.pq`. Requires Python 3.11+.

## Input format

Six columns are required, in any order:

| Column | Type |
|---|---|
| `timestamp` | datetime |
| `open`, `high`, `low`, `close` | float |
| `volume` | integer |

- **Column names are normalised** — `Timestamp`, ` OPEN ` and `Volume` all work.
- **Extra columns are ignored.**
- **Missing columns are reported, not fatal.** `missing_columns` fails and the rules needing those
  columns skip; you still get a full report.
- **Unparseable values become null** and are reported by `null_values`. One bad row does not discard
  the rest of the column.
- The ticker is inferred from the filename when possible (`AAPL_1min.csv` → `AAPL`).

## Sample output

```
========================================================================
  MarketCheck Report  ⚠  WARN
========================================================================
  File:   AAPL_1min.csv
  Ticker: AAPL
  Rows:   18
  Range:  2024-01-02 09:30:00 → 2024-01-02 09:49:00

  Rules run: 14  |  Passed: 12  |  Warned: 2  |  Failed: 0  |  Skipped: 0
------------------------------------------------------------------------
  ⚠ [WARN] Volume Anomaly (numerical.volume_anomaly)
         1 bar(s) have volume exceeding 10x the 20-bar rolling median and may
         warrant review (first at index 13: volume 900000 vs median 1000, 900x).
         Affected rows: 1

  ⚠ [WARN] Gaps Within Session (temporal.gaps_within_session)
         1 gap(s) within sessions, 2 missing 1min bar(s) in total (largest 2;
         first after 2024-01-02 09:36:00, 2 bar(s) absent).
         Affected rows: 1

  ✓ [PASS] Timezone Inconsistency (temporal.timezone_inconsistency)
         Timestamps are consistent with ET wall-clock.
========================================================================
```

`-f json` emits the same content plus per-finding `details` — row indices, offending values, and the
thresholds used, so any finding can be audited.

## Options

| Option | Description |
|---|---|
| `--format`, `-f` | `text` (default) or `json` |
| `--strict` | Treat warnings as failures (affects exit code only) |
| `--output`, `-o` | Write the report to a file instead of stdout |
| `--config`, `-c` | Path to a TOML config file |

## Exit codes

Validation failure and tool failure are distinct, so CI can tell "the data is bad" from "the tool
could not run".

| Code | Meaning |
|---|---|
| `0` | No failures. Warnings alone pass unless `--strict`. |
| `1` | Validation failed — a rule reported `fail`, or a warning under `--strict`. |
| `2` | Tool error — file missing, unsupported extension, unreadable content, or bad config. |

Warnings do not fail by default because many findings need human judgement: a genuine trading halt
looks like a data gap, and a real biotech collapse looks like a bad tick.

## Four outcomes, not two

| Status | Meaning |
|---|---|
| `pass` | The rule inspected the data and found nothing wrong. |
| `warn` | A finding that needs human judgement. |
| `fail` | A finding that makes the data unusable as-is. |
| `skip` | The rule could **not** evaluate the data, so nothing was verified. |

A rule skips when a prerequisite is missing — no `volume` column for `volume_anomaly`, no
`timestamp` for temporal rules, an uninferable bar frequency, too few rows, or the rule being
disabled in config. Skips never worsen the overall status and are never counted as passes.

Consequently a file with **no rows** reports `warn`, not `pass`; and if no rule produces a verdict,
the run reports `warn` and says why rather than giving an empty all-clear.

## Timestamps

Temporal checks measure against NYSE session hours, so timestamps must mean **ET wall-clock**.

- **With an offset** (`2024-01-02T09:30:00-05:00`, or trailing `Z`) — converted to ET, then stored
  without a zone. Conversion goes through a real timezone, so DST is handled: in July `13:30Z`
  becomes `09:30`.
- **Naive** (`2024-01-02 09:30:00`) — assumed to already be ET. `timezone_inconsistency` checks that
  assumption rather than trusting it: if bars only line up with the session after a uniform shift,
  the file is flagged.

## What it checks

14 rules. `critical` means the data is unusable, `warning` usable but questionable, `info`
noteworthy but benign.

**Structural** — is the file shaped like OHLCV data?

| Rule | Severity |
|---|---|
| `structural.missing_columns` | critical |
| `structural.invalid_dtypes` | critical |
| `structural.duplicate_timestamps` | critical |
| `structural.null_values` | warning |
| `structural.unsorted_timestamps` | warning |

**Temporal** — is the time axis complete and coherent?

| Rule | Severity |
|---|---|
| `temporal.timezone_inconsistency` | critical |
| `temporal.missing_sessions` | warning |
| `temporal.gaps_within_session` | warning |
| `temporal.outside_trading_hours` | info |

**Numerical** — are the numbers internally consistent and plausible?

| Rule | Severity |
|---|---|
| `numerical.ohlc_range_violation` | critical |
| `numerical.impossible_values` | critical |
| `numerical.volume_anomaly` | warning |
| `numerical.suspicious_price_jump` | warning |

**Financial** — does the data reflect market reality?

| Rule | Severity |
|---|---|
| `financial.corporate_action_discontinuity` | info |

## Configuration

Flat TOML. Keys map one-to-one onto settings, and **unknown keys are rejected** — a silently dropped
typo would leave you believing you had configured something you had not.

```toml
# marketcheck.toml
volume_anomaly_multiplier = 15.0   # raise for a vendor reporting round lots
gap_min_missing_bars      = 2      # ignore single-bar holes on illiquid names
disabled_rules            = ["temporal.outside_trading_hours"]
```

```bash
marketcheck AAPL_1min.csv --config marketcheck.toml
```

| Setting | Default | Effect |
|---|---|---|
| `max_rows_in_details` | `50` | Cap on rows listed per finding |
| `volume_anomaly_multiplier` | `10.0` | Multiple of the rolling median that flags a bar |
| `volume_anomaly_window` | `20` | Rolling-median window, in bars |
| `price_jump_intraday_threshold` | `0.20` | Fractional move flagged within a session |
| `price_jump_overnight_threshold` | `0.50` | Fractional move flagged across a session boundary |
| `split_ratio_tolerance` | `0.02` | Tolerance when matching a split ratio |
| `gap_min_missing_bars` | `1` | Smallest gap reported |
| `gap_frequency_dominance` | `0.5` | Share of intervals the modal bar spacing must exceed |
| `enabled_categories` | all | Subset of `structural`, `temporal`, `numerical`, `financial` |
| `disabled_rules` | none | Full rule IDs to skip (still reported, as skips) |

`--format`, `--output`, and `--strict` are command-line only and rejected in config files: they
describe how the tool was invoked, so a committed config cannot redirect someone else's output.

## What it deliberately does not cover

Knowing a tool's blind spots is part of trusting it.

- **Dividends are undetectable.** A dividend drop is 0.1–3% of price, indistinguishable from
  ordinary movement. `corporate_action_discontinuity` detects **splits only**, despite its name.
- **Corporate actions are inferred, not confirmed.** No external data is consulted, so findings say
  "consistent with a 2:1 split", never "a 2:1 split occurred".
- **Edge truncation is invisible.** Neither `missing_sessions` nor `gaps_within_session` can tell
  that data begins a week late or that a session ended early — only that something is missing
  *between* two things that are present.
- **Single-bar "wick" spikes go undetected.** A bar with `high=10000` and `close≈100` is
  geometrically valid and does not move close-to-close returns, so nothing catches it.
- **Gaps must be the minority.** Bar frequency is the most common spacing, so data missing more than
  half its bars will have the *gap* spacing mistaken for the real grid.
- **No tick data.** Rules assume a fixed bar grid; irregular data is skipped, not flagged.
- **Timezone checking needs 30-minute bars or finer.** The check requires 10+ distinct times of
  day, so it runs on 1min–30min data and skips on hourly and daily. With few distinct times a
  uniform shift is indistinguishable from the data simply being coarse — daily bars stamped
  `00:00` would be "fixed" into the session by a `+10:00` shift, so guessing would flag every
  daily file as critically broken.
- **A declared offset is visible, but not which one.** Timestamps are normalised on read, so the
  tool knows a source carried a timezone, not that it said `-05:00` specifically.

## Development

```bash
uv sync
pytest              # 443 passed
ruff check .
mypy src            # strict
```
