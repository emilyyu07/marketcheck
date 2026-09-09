"""
Timezone normalization utilities for market data timestamps.
Provides NYSE market awareness (i.e. how to reason about timezones) to validators.
"""

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

    # localize  (takes no timezone into account, declares it in given timezone)
    # replace_time_zone is used to set the timezone without changing the underlying timestamp values
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

    # normalize (takes a timezone-aware dattime and coverts into different timezone)
    # convert_time_zone is used to convert the timestamp values to UTC,
    # adjusting clock values accordingly
    return df.with_columns(
        pl.col(column).dt.convert_time_zone("UTC")
    )
