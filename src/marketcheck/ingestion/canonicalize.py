"""Canonicalize raw DataFrames into CanonicalDataset instances."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from marketcheck.ingestion.schema import coerce_dtypes
from marketcheck.models.dataset import CanonicalDataset


def to_canonical(df: pl.DataFrame, source_path: str | Path) -> CanonicalDataset:
    """Transform a raw DataFrame into a CanonicalDataset.

    Applies dtype coercion, extracts metadata, and wraps in the
    canonical model.

    Args:
        df: Raw OHLCV DataFrame (already loaded from file).
        source_path: Path the data was loaded from.

    Returns:
        A CanonicalDataset ready for validation.
    """
    df = coerce_dtypes(df)

    # Extract metadata
    row_count = len(df)
    start_time = None
    end_time = None
    if row_count > 0 and "timestamp" in df.columns:
        ts_col = df["timestamp"]
        start_time = ts_col.min()
        end_time = ts_col.max()

    # Attempt to extract ticker from filename
    ticker = Path(source_path).stem.split("_")[0].upper()

    return CanonicalDataset(
        df=df,
        source_path=str(source_path),
        ticker=ticker,
        row_count=row_count,
        start_time=start_time,
        end_time=end_time,
    )
