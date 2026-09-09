# MarketCheck

A command-line tool that inspects historical US-equity OHLCV market data and reports whether that
data is structurally and financially trustworthy enough to use for research or backtesting.

The guiding principle is that **a check which did not run is never reported as a pass.** A report
distinguishes "this was verified and is fine" from "this could not be verified", because a tool you
consult to decide whether to trust your data is worthless if it overstates what it checked.

## Quick Start

```bash
# Install
uv sync

# Run a validation
marketcheck AAPL_1min.csv

# JSON output
marketcheck AAPL_1min.csv --format json

# Fail the build on warnings too
marketcheck AAPL_1min.csv --strict
```

Accepts `.csv`, `.parquet`, and `.pq` files.

## Options

| Option | Description |
|---|---|
| `--format`, `-f` | `text` (default) or `json` |
| `--strict` | Treat warnings as failures (affects the exit code) |
| `--output`, `-o` | Write the report to a file instead of stdout |
| `--config`, `-c` | Path to a config file |

## Configuration

Thresholds can be tuned with a flat TOML file. Every key maps one-to-one onto a
documented setting, and **unknown keys are rejected** rather than ignored — a silently
dropped typo would leave you believing you had configured something you had not.

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
| `price_jump_intraday_threshold` | `0.20` | Move flagged within a session |
| `price_jump_overnight_threshold` | `0.50` | Move flagged across a session boundary |
| `split_ratio_tolerance` | `0.02` | Tolerance when matching a split ratio |
| `gap_min_missing_bars` | `1` | Smallest gap reported |
| `gap_frequency_dominance` | `0.5` | Share of intervals the modal spacing must exceed |
| `enabled_categories` | all | Restrict to given categories |
| `disabled_rules` | none | Rule IDs to skip (still reported, as skips) |

`--format`, `--output`, and `--strict` are command-line only: they describe how the tool
was invoked, not how validation should behave, so a committed config file cannot silently
redirect someone else's output.

## Exit codes

Validation failure and tool failure are deliberately distinct, so CI can tell "the data is bad"
from "the tool could not run".

| Code | Meaning |
|---|---|
| `0` | No failures. Warnings alone do not fail unless `--strict` is passed. |
| `1` | Validation failed — at least one rule reported `FAIL` (or a warning under `--strict`). |
| `2` | Tool error — file missing, unsupported extension, or unreadable content. |

Warnings do not fail the build by default because many findings need human judgement: a genuine
trading halt looks like a data gap, and a real biotech collapse looks like a bad tick.

## Passed, or simply not checked?

Every report separates four outcomes, because the difference between "verified fine" and "not
verified" is the whole point of the tool.

| Status | Meaning |
|---|---|
| `pass` | The rule inspected the data and found nothing wrong. |
| `warn` | A finding that needs human judgement. |
| `fail` | A finding that makes the data unusable as-is. |
| `skip` | The rule could **not** evaluate the data, so nothing was verified. |

A rule skips when a prerequisite is absent — no `volume` column for `volume_anomaly`, no
`timestamp` for the temporal rules, a bar frequency that cannot be inferred, fewer rows than the
rule needs, or the rule being disabled or not yet implemented. Skips never worsen the overall
status, but they are never counted as passes either.

Two consequences worth knowing:

- A file with **no rows** reports `warn`, not `pass`. Nothing about it could be validated.
- If **no** rule produces a verdict, the run reports `warn` and says so, rather than presenting an
  empty all-clear.

## What it checks

14 rules across four categories. Severity reflects how bad a finding would be: `critical` means the
data is unusable, `warning` means usable but questionable, `info` means noteworthy but benign.

**Structural** — is the file even shaped like OHLCV data?

| Rule | Severity |
|---|---|
| `missing_columns` | critical |
| `invalid_dtypes` | critical |
| `duplicate_timestamps` | critical |
| `null_values` | warning |
| `unsorted_timestamps` | warning |

**Temporal** — is the time axis complete and coherent?

| Rule | Severity |
|---|---|
| `timezone_inconsistency` | critical |
| `missing_sessions` | warning |
| `gaps_within_session` | warning |
| `outside_trading_hours` | info |

**Numerical** — are the numbers internally consistent and plausible?

| Rule | Severity |
|---|---|
| `ohlc_range_violation` | critical |
| `impossible_values` | critical |
| `volume_anomaly` | warning |
| `suspicious_price_jump` | warning |

**Financial** — does the data reflect market reality?

| Rule | Severity |
|---|---|
| `corporate_action_discontinuity` | info |

## What it deliberately does not cover

Stated plainly, because knowing a tool's blind spots is part of trusting it.

- **Dividends are undetectable.** A dividend drop is 0.1–3% of price, indistinguishable from
  ordinary movement. `corporate_action_discontinuity` detects **splits only**, despite its name.
- **Corporate actions are inferred, not confirmed.** No external data is consulted, so findings say
  "consistent with a 2:1 split", never "a 2:1 split occurred".
- **Edge truncation is invisible.** Neither `missing_sessions` nor `gaps_within_session` can tell
  that a dataset begins a week late or that a session ends early — only that something is missing
  *between* two things that are present.
- **Single-bar "wick" spikes go undetected.** A bar with `high=10000` and `close≈100` is
  geometrically valid and does not move close-to-close returns, so nothing currently catches it.
- **Gaps must be the minority.** Bar frequency is inferred as the most common spacing, so data
  missing more than half its bars will have the *gap* spacing mistaken for the real grid.
- **No tick data.** Rules assume a fixed bar grid; irregular data is skipped rather than flagged.

## Development

```bash
uv sync
pytest
ruff check .
mypy src
```
