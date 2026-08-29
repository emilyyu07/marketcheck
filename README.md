# MarketCheck

A command-line tool that inspects historical US-equity OHLCV market data and reports whether that
data is structurally and financially trustworthy enough to use for research or backtesting.

## Quick Start

```bash
# Install
uv sync

# Run a validation
marketcheck validate AAPL_1min.csv

# JSON output
marketcheck validate AAPL_1min.csv --format json
```

## Features

- **Structural checks** — duplicate timestamps, required columns, sort order, data types
- **Temporal checks** — missing sessions, gaps within sessions, timezone consistency
- **Numerical checks** — OHLC range violations, volume anomalies, price jump detection
- **Financial checks** — corporate-action discontinuity detection

## Development

```bash
uv sync
pytest
ruff check .
```
