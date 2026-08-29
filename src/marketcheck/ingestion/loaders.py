"""File loaders for CSV and Parquet data sources."""

from __future__ import annotations

from pathlib import Path

import polars as pl


def load_csv(path: str | Path) -> pl.DataFrame:
    """Load a CSV file into a Polars DataFrame.

    Args:
        path: Path to the CSV file.

    Returns:
        Raw DataFrame with data as-is from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be parsed as CSV.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    try:
        return pl.read_csv(path, try_parse_dates=True)
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV file {path}: {exc}") from exc


def load_parquet(path: str | Path) -> pl.DataFrame:
    """Load a Parquet file into a Polars DataFrame.

    Args:
        path: Path to the Parquet file.

    Returns:
        Raw DataFrame with data as-is from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be parsed as Parquet.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    try:
        return pl.read_parquet(path)
    except Exception as exc:
        raise ValueError(f"Failed to parse Parquet file {path}: {exc}") from exc
