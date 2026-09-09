"""Schema definitions and dtype coercion for OHLCV data."""

from __future__ import annotations

import polars as pl
from polars._typing import PolarsDataType

# Canonical column names expected in every dataset.
REQUIRED_COLUMNS: list[str] = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

# Expected Polars dtypes after coercion.
# Values are a mix of dtype classes (pl.Float64) and parameterised instances
# (pl.Datetime("us")); PolarsDataType is polars' own union of the two.
EXPECTED_DTYPES: dict[str, PolarsDataType] = {
    "timestamp": pl.Datetime("us"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Int64,
}


def coerce_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    """Coerce DataFrame columns to expected OHLCV dtypes.

    Normalizes column names to lowercase and casts whichever expected columns are
    present.

    **Deliberately tolerant of missing columns.** An earlier version raised
    `ValueError` here, which defeated the architecture: ingestion aborted before
    validation ran, so the `structural.missing_columns` rule -- whose entire
    purpose is to report absent columns -- could never fire through the CLI, and
    the eight rules that guard with "MissingColumns owns absent columns" were
    likewise unreachable. The user saw a raw traceback instead of a report naming
    what was wrong.

    Reporting missing columns is a validation concern, not an ingestion concern.
    Ingestion's job is to normalise what it was given; judging completeness
    belongs to the rules.

    Args:
        df: Raw input DataFrame.

    Returns:
        DataFrame with canonical column names, and expected dtypes applied to
        whichever expected columns exist.
    """
    # Normalize column names to lowercase
    df = df.rename({col: col.strip().lower() for col in df.columns})

    # Cast whichever expected columns are present. Absent columns are left to
    # structural.missing_columns to report.
    cast_exprs: list[pl.Expr] = []
    for col_name, dtype in EXPECTED_DTYPES.items():
        if col_name in df.columns:
            cast_exprs.append(pl.col(col_name).cast(dtype, strict=False))

    if cast_exprs:
        df = df.with_columns(cast_exprs)

    return df
