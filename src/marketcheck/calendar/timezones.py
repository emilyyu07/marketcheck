"""Timezone normalization utilities for market data timestamps."""

from __future__ import annotations

from zoneinfo import ZoneInfo

import polars as pl

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def localize_to_et(df: pl.DataFrame, column: str = "timestamp") -> pl.DataFrame:
    """Localize a naive timestamp column to US/Eastern.

    Args:
        df: DataFrame with a timestamp column.
        column: Name of the timestamp column.

    Returns:
        DataFrame with the timestamp column timezone-aware in ET.
    """
    return df.with_columns(
        pl.col(column).dt.replace_time_zone("America/New_York")
    )


def normalize_to_utc(df: pl.DataFrame, column: str = "timestamp") -> pl.DataFrame:
    """Convert a timezone-aware timestamp column to UTC.

    Args:
        df: DataFrame with a timezone-aware timestamp column.
        column: Name of the timestamp column.

    Returns:
        DataFrame with the timestamp column in UTC.
    """
    return df.with_columns(
        pl.col(column).dt.convert_time_zone("UTC")
    )
