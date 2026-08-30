"""Schema definitions and dtype coercion for OHLCV data."""

from __future__ import annotations

import polars as pl

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
EXPECTED_DTYPES: dict[str, pl.DataType] = {
    "timestamp": pl.Datetime("us"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Int64,
}


def coerce_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    """Coerce DataFrame columns to expected OHLCV dtypes.

    Normalizes column names to lowercase and casts numeric columns.
    Raises if required columns are missing.

    Args:
        df: Raw input DataFrame.

    Returns:
        DataFrame with canonical column names and dtypes.

    Raises:
        ValueError: If required columns are missing.
    """
    # Normalize column names to lowercase
    df = df.rename({col: col.strip().lower() for col in df.columns})

    # raise if any required columns are missing (each rule depends on all columns existing)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Cast numeric columns
    cast_exprs: list[pl.Expr] = []
    for col_name, dtype in EXPECTED_DTYPES.items():
        if col_name in df.columns:
            cast_exprs.append(pl.col(col_name).cast(dtype, strict=False))

    if cast_exprs:
        df = df.with_columns(cast_exprs)

    return df
