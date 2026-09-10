"""Canonical dataset wrapper for validated OHLCV DataFrames."""

from __future__ import annotations

from datetime import datetime

import polars as pl
from pydantic import BaseModel, ConfigDict


class CanonicalDataset(BaseModel):
    """A validated, normalized OHLCV dataset ready for validation rules.

    Wraps a Polars DataFrame that has been through schema coercion and
    column normalization. Carries metadata about the dataset for use by
    validators and reporting.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    df: pl.DataFrame
    """The OHLCV data with canonical column names and dtypes."""

    source_path: str
    """Original file path the data was loaded from."""

    ticker: str = ""
    """Ticker symbol, if known (extracted from filename or metadata)."""

    row_count: int = 0
    """Number of rows in the dataset."""

    start_time: datetime | None = None
    """Earliest timestamp in the dataset."""

    end_time: datetime | None = None
    """Latest timestamp in the dataset."""

    source_timezone: str | None = None
    """What timezone information the SOURCE timestamps carried, before coercion.

    Coercion normalises timestamps to naive ET wall-clock, which destroys the
    original representation, so this is recorded during ingestion for
    `temporal.timezone_inconsistency` to report on.

    One of: `None` (no timestamp column), a timezone name such as `"UTC"` (the
    source was timezone-aware), `"naive"` (no timezone information), `"mixed"`
    (offset-bearing and offset-free values in the same column, which cannot be
    parsed as one type and silently becomes nulls), or `"unparsed"`.
    """
