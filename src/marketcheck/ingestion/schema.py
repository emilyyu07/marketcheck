"""Schema definitions and dtype coercion for OHLCV data."""

from __future__ import annotations

import polars as pl
from polars._typing import PolarsDataType

# Market data is validated against NYSE session hours, so ET wall-clock is the
# canonical frame of reference for every temporal rule.
MARKET_TIMEZONE = "America/New_York"

# Values recorded on CanonicalDataset.source_timezone.
TZ_NAIVE = "naive"
TZ_MIXED = "mixed"
TZ_UNPARSED = "unparsed"

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


def describe_timestamp_timezone(df: pl.DataFrame) -> str | None:
    """Describe what timezone information the SOURCE timestamps carried.

    Must be called on the raw DataFrame, before `coerce_dtypes` normalises it --
    coercion deliberately produces naive ET wall-clock, at which point the
    original representation is no longer recoverable.

    Returns:
        `None` if there is no timestamp column; the timezone name (e.g. `"UTC"`)
        if the source was timezone-aware; `"naive"` if it carried no timezone;
        `"mixed"` if the column mixes offset-bearing and offset-free values (which
        polars cannot parse as a single type, leaving the column as strings); or
        `"unparsed"` for a string column that is not a timezone mixture.
    """
    if "timestamp" not in df.columns:
        return None

    dtype = df.schema["timestamp"]

    if isinstance(dtype, pl.Datetime):
        # A single consistent representation. Note that a source mixing DIFFERENT
        # offsets (+00:00 and -05:00) still lands here, correctly: polars resolves
        # them to the same instant scale, so they are consistent in meaning.
        return dtype.time_zone if dtype.time_zone is not None else TZ_NAIVE

    if dtype == pl.String:
        # polars left the column as strings, so it could not parse it as one type.
        # The case worth naming is a genuine mixture of offset-bearing and
        # offset-free values, because those rows silently become nulls in coercion.
        values = df.get_column("timestamp").drop_nulls()
        if values.len() == 0:
            return TZ_UNPARSED

        # Only DATE-LIKE values participate. Without this, a column of
        # ["2024-01-02T09:30:00-05:00", "garbage"] would look like an aware/naive
        # mixture and be reported as a CRITICAL timezone inconsistency, when the
        # real problem is one unparseable value and nothing to do with timezones.
        date_like = values.filter(values.str.contains(r"^\d{4}-\d{2}-\d{2}"))
        if date_like.len() == 0:
            return TZ_UNPARSED

        # An offset suffix (+HH:MM / -HH:MM) or a trailing Z.
        has_offset = date_like.str.contains(r"(?:[+-]\d{2}:?\d{2}|Z)$")
        offset_count = int(has_offset.sum())

        if offset_count == 0:
            return TZ_NAIVE
        if offset_count == date_like.len():
            # Every timestamp declares an offset. Coercion normalises these to ET
            # wall-clock, so report it the same way an already-parsed aware column
            # is reported.
            return "UTC"
        return TZ_MIXED

    return TZ_UNPARSED


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

    # Step 1: parse an unparsed timestamp column.
    #
    # `cast(Datetime)` is wrong here twice over. It is deprecated (removed in polars
    # 2.0), and it is DESTRUCTIVE: given ["2024-01-02 09:30:00", "not-a-date"] it
    # returns [None, None], nulling the valid row too. One malformed row should not
    # erase every timestamp in the file.
    #
    # The parse runs EAGERLY on the Series rather than as an expression. polars
    # refuses format inference inside an expression when a timezone appears in the
    # data ("perform the operation eagerly on a Series instead"), which would raise
    # a ComputeError on any offset-bearing string column and cost the user their
    # whole report. Eager parsing handles naive, aware, and partially-malformed
    # input alike.
    if "timestamp" in df.columns and df.schema["timestamp"] == pl.String:
        series = df.get_column("timestamp")
        try:
            df = df.with_columns(series.str.to_datetime(time_unit="us", strict=False))
        except Exception:
            # Nothing in the column resembles a datetime, so no format can be
            # inferred. Null the column rather than raising: MissingColumns,
            # NullValues, and timezone_inconsistency can all still report on the
            # file, whereas an exception would deny the user any report at all.
            df = df.with_columns(
                pl.Series(
                    "timestamp", [None] * df.height, dtype=pl.Datetime("us")
                )
            )

    # Step 2: convert a timezone-aware column to ET WALL-CLOCK before dropping the
    # zone. Casting straight to a naive dtype keeps whatever clock values the aware
    # column held -- polars normalises offsets to UTC on read, so a correct
    # `09:30-05:00` bar would become a naive `14:30`, and every rule comparing
    # against NYSE session hours would then be five hours wrong. Reading a declared
    # offset correctly is not repairing the data; discarding it is misreading.
    if "timestamp" in df.columns:
        ts_dtype = df.schema["timestamp"]
        if isinstance(ts_dtype, pl.Datetime) and ts_dtype.time_zone is not None:
            df = df.with_columns(
                pl.col("timestamp")
                .dt.convert_time_zone(MARKET_TIMEZONE)
                .dt.replace_time_zone(None)
            )

    # Step 3: cast the remaining expected columns. Absent columns are left to
    # structural.missing_columns to report. `timestamp` is skipped once it is a
    # Datetime, since steps 1 and 2 already produced the canonical form.
    cast_exprs: list[pl.Expr] = []
    for col_name, dtype in EXPECTED_DTYPES.items():
        if col_name not in df.columns:
            continue
        if col_name == "timestamp" and isinstance(df.schema["timestamp"], pl.Datetime):
            continue
        cast_exprs.append(pl.col(col_name).cast(dtype, strict=False))

    if cast_exprs:
        df = df.with_columns(cast_exprs)

    return df
