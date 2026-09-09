"""Tests for the ingestion module."""

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from marketcheck.ingestion.canonicalize import to_canonical
from marketcheck.ingestion.loaders import load_csv, load_parquet
from marketcheck.ingestion.schema import (
    REQUIRED_COLUMNS,
    coerce_dtypes,
    describe_timestamp_timezone,
)


class TestLoadCsv:
    def test_loads_valid_csv(self, sample_csv_path: Path) -> None:
        df = load_csv(sample_csv_path)
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 10

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        import pytest

        with pytest.raises(FileNotFoundError):
            load_csv(tmp_path / "nonexistent.csv")


class TestLoadParquet:
    def test_loads_valid_parquet(self, sample_parquet_path: Path) -> None:
        df = load_parquet(sample_parquet_path)
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 10


class TestCoerceDtypes:
    def test_coerces_valid_df(self, sample_ohlcv_df: pl.DataFrame) -> None:
        result = coerce_dtypes(sample_ohlcv_df)
        assert all(col in result.columns for col in REQUIRED_COLUMNS)


class TestToCanonical:
    def test_produces_canonical_dataset(self, sample_ohlcv_df: pl.DataFrame) -> None:
        dataset = to_canonical(sample_ohlcv_df, source_path="test.csv")
        assert dataset.row_count == 10
        assert dataset.source_path == "test.csv"


class TestTimezoneCoercion:
    """Timezone-aware sources must be converted to ET wall-clock, not merely stripped.

    Casting an aware column straight to a naive dtype keeps whatever clock values it
    held, and polars normalises offsets to UTC on read — so a correct `09:30-05:00`
    bar became a naive `14:30`, and every rule comparing against NYSE session hours
    was then five hours wrong. Measured before the fix: 300 of 390 bars in a valid
    session were reported outside trading hours.
    """

    @staticmethod
    def _csv(tmp_path: Path, timestamps: list[str]) -> Path:
        path = tmp_path / "tz.csv"
        path.write_text(
            "timestamp,open,high,low,close,volume\n"
            + "\n".join(f"{ts},100,101,99,100,1000" for ts in timestamps)
            + "\n",
            encoding="utf-8",
        )
        return path

    def test_et_offset_preserves_wall_clock(self, tmp_path: Path) -> None:
        """09:30 declared as -05:00 must stay 09:30, not become 14:30."""
        path = self._csv(
            tmp_path, ["2024-01-02T09:30:00-05:00", "2024-01-02T09:31:00-05:00"]
        )
        dataset = to_canonical(load_csv(path), source_path=path)

        assert dataset.df["timestamp"][0] == datetime(2024, 1, 2, 9, 30)

    def test_utc_offset_converted_to_et(self, tmp_path: Path) -> None:
        """14:30 UTC is 09:30 ET in January; the ET clock value is what rules need."""
        path = self._csv(
            tmp_path, ["2024-01-02T14:30:00+00:00", "2024-01-02T14:31:00+00:00"]
        )
        dataset = to_canonical(load_csv(path), source_path=path)

        assert dataset.df["timestamp"][0] == datetime(2024, 1, 2, 9, 30)

    def test_zulu_suffix_converted_to_et(self, tmp_path: Path) -> None:
        path = self._csv(tmp_path, ["2024-01-02T14:30:00Z", "2024-01-02T14:31:00Z"])
        dataset = to_canonical(load_csv(path), source_path=path)

        assert dataset.df["timestamp"][0] == datetime(2024, 1, 2, 9, 30)

    def test_dst_offset_converted_correctly(self, tmp_path: Path) -> None:
        """In July, ET is -04:00, so 13:30 UTC is 09:30 ET. A fixed -5 would be wrong,
        which is why the conversion uses a real timezone rather than an offset."""
        path = self._csv(
            tmp_path, ["2024-07-02T13:30:00+00:00", "2024-07-02T13:31:00+00:00"]
        )
        dataset = to_canonical(load_csv(path), source_path=path)

        assert dataset.df["timestamp"][0] == datetime(2024, 7, 2, 9, 30)

    def test_naive_timestamps_left_untouched(self, tmp_path: Path) -> None:
        """A naive source is already assumed to be ET wall-clock."""
        path = self._csv(tmp_path, ["2024-01-02 09:30:00", "2024-01-02 09:31:00"])
        dataset = to_canonical(load_csv(path), source_path=path)

        assert dataset.df["timestamp"][0] == datetime(2024, 1, 2, 9, 30)

    def test_result_dtype_is_naive(self, tmp_path: Path) -> None:
        """Downstream rules compare naive-to-naive, so the canonical dtype must stay
        timezone-free after conversion."""
        path = self._csv(tmp_path, ["2024-01-02T09:30:00-05:00"])
        dataset = to_canonical(load_csv(path), source_path=path)

        dtype = dataset.df.schema["timestamp"]
        assert isinstance(dtype, pl.Datetime)
        assert dtype.time_zone is None


class TestTimezoneProvenance:
    """`source_timezone` records what the source declared, before coercion erases it."""

    @staticmethod
    def _df(timestamps: list[str]) -> pl.DataFrame:
        return pl.DataFrame({"timestamp": timestamps})

    def test_naive_source_recorded(self, tmp_path: Path) -> None:
        path = TestTimezoneCoercion._csv(tmp_path, ["2024-01-02 09:30:00"])
        assert to_canonical(load_csv(path), source_path=path).source_timezone == "naive"

    def test_aware_source_recorded(self, tmp_path: Path) -> None:
        """polars normalises offsets to UTC on read, so the specific declared offset
        is not recoverable — only that the source was aware."""
        path = TestTimezoneCoercion._csv(tmp_path, ["2024-01-02T09:30:00-05:00"])
        assert to_canonical(load_csv(path), source_path=path).source_timezone == "UTC"

    def test_mixed_aware_and_naive_recorded(self, tmp_path: Path) -> None:
        """This mixture cannot be parsed as one type, so the rows silently become
        nulls. Recording it lets timezone_inconsistency name the cause instead of
        leaving the user with unexplained nulls."""
        path = TestTimezoneCoercion._csv(
            tmp_path, ["2024-01-02 09:30:00", "2024-01-02T14:31:00+00:00"]
        )
        assert to_canonical(load_csv(path), source_path=path).source_timezone == "mixed"

    def test_mixed_offsets_are_not_mixed(self, tmp_path: Path) -> None:
        """Different offsets are NOT an inconsistency: polars resolves them to the
        same instant scale, so they agree in meaning."""
        path = TestTimezoneCoercion._csv(
            tmp_path, ["2024-01-02T14:30:00+00:00", "2024-01-02T09:31:00-05:00"]
        )
        assert to_canonical(load_csv(path), source_path=path).source_timezone == "UTC"

    def test_unparseable_source_recorded(self) -> None:
        df = self._df(["not-a-date", "also-not"])
        assert describe_timestamp_timezone(df) == "unparsed"

    def test_no_timestamp_column(self) -> None:
        assert describe_timestamp_timezone(pl.DataFrame({"close": [1.0]})) is None

    def test_describe_must_run_before_coercion(self, tmp_path: Path) -> None:
        """Coercion produces naive ET, so provenance is unrecoverable afterwards —
        which is why to_canonical records it first."""
        path = TestTimezoneCoercion._csv(tmp_path, ["2024-01-02T09:30:00-05:00"])
        raw = load_csv(path)

        assert describe_timestamp_timezone(raw) == "UTC"
        assert describe_timestamp_timezone(coerce_dtypes(raw)) == "naive"


class TestTimestampParseRobustness:
    """Parsing must never destroy valid rows, and must never raise.

    `cast(Datetime)` was wrong twice over: deprecated (removed in polars 2.0), and
    destructive — given ["2024-01-02 09:30:00", "not-a-date"] it returned
    [None, None], nulling the valid row. Replacing it with an expression-based
    `str.to_datetime` then introduced the opposite failure: polars refuses format
    inference inside an expression when a timezone appears in the data, raising a
    ComputeError and costing the user their entire report. The parse therefore runs
    eagerly on the Series, with a null fallback for wholly unparseable input.
    """

    @staticmethod
    def _coerce(values: list[str]) -> list[datetime | None]:
        df = pl.DataFrame({"timestamp": values})
        return coerce_dtypes(df)["timestamp"].to_list()

    def test_one_bad_row_does_not_null_the_good_rows(self) -> None:
        result = self._coerce(
            ["2024-01-02 09:30:00", "not-a-date", "2024-01-02 09:32:00"]
        )

        assert result[0] == datetime(2024, 1, 2, 9, 30)
        assert result[1] is None
        assert result[2] == datetime(2024, 1, 2, 9, 32)

    def test_no_deprecation_warning(self) -> None:
        """Guards against reintroducing the String->Datetime cast."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            self._coerce(["2024-01-02 09:30:00"])

    @pytest.mark.parametrize(
        "values",
        [
            ["2024-01-02T09:30:00-05:00", "2024-01-02T09:31:00-05:00"],
            ["2024-01-02 09:30:00", "2024-01-02T14:31:00+00:00"],
            ["2024-01-02 09:30:00", "2024-01-02 09:31:00"],
            ["nope", "also-nope"],
            ["2024-01-02T09:30:00-05:00", "nope"],
            ["2024-01-02 09:30:00", "nope"],
            [],
        ],
    )
    def test_never_raises(self, values: list[str]) -> None:
        """A file that cannot be parsed must still yield a report. Raising here
        would deny the user any validation output at all."""
        df = pl.DataFrame({"timestamp": values}, schema={"timestamp": pl.String})
        result = coerce_dtypes(df)

        assert isinstance(result.schema["timestamp"], pl.Datetime)

    def test_wholly_unparseable_column_becomes_null(self) -> None:
        """Nulling lets MissingColumns, NullValues and timezone_inconsistency all
        still report on the file."""
        assert self._coerce(["nope", "also-nope"]) == [None, None]

    def test_aware_strings_are_converted_to_et(self) -> None:
        """String columns carrying offsets still reach ET wall-clock."""
        assert self._coerce(["2024-01-02T09:30:00-05:00"]) == [
            datetime(2024, 1, 2, 9, 30)
        ]


class TestProvenanceAccuracy:
    """`mixed` drives a CRITICAL finding, so it must not be claimed loosely."""

    @staticmethod
    def _describe(values: list[str]) -> str | None:
        return describe_timestamp_timezone(pl.DataFrame({"timestamp": values}))

    def test_aware_plus_garbage_is_not_reported_as_mixed(self) -> None:
        """Only date-like values participate. Otherwise one unparseable value would
        be mistaken for a naive timestamp, and the rule would report a CRITICAL
        "mixes aware and naive values" on a file with no naive timestamp at all."""
        assert self._describe(["2024-01-02T09:30:00-05:00", "nope"]) == "UTC"

    def test_naive_plus_garbage_is_not_reported_as_mixed(self) -> None:
        assert self._describe(["2024-01-02 09:30:00", "nope"]) == "naive"

    def test_genuine_mixture_is_reported_as_mixed(self) -> None:
        assert (
            self._describe(["2024-01-02 09:30:00", "2024-01-02T14:31:00+00:00"])
            == "mixed"
        )

    def test_all_aware_strings_reported_as_aware(self) -> None:
        assert self._describe(["2024-01-02T09:30:00-05:00"]) == "UTC"

    def test_all_naive_strings_reported_as_naive(self) -> None:
        assert self._describe(["2024-01-02 09:30:00"]) == "naive"

    def test_all_garbage_reported_as_unparsed(self) -> None:
        assert self._describe(["nope", "also-nope"]) == "unparsed"

    def test_zulu_suffix_counts_as_aware(self) -> None:
        assert self._describe(["2024-01-02T14:30:00Z"]) == "UTC"
