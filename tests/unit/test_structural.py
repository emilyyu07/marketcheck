"""Tests for structural validation rules."""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from marketcheck.models.config import ValidationConfig
from marketcheck.models.dataset import CanonicalDataset
from marketcheck.models.enums import Category, Severity, Status
from marketcheck.validators import REGISTRY
from marketcheck.validators.base import RuleContext
from marketcheck.validators.structural import (
    DuplicateTimestamps,
    InvalidDtypes,
    MissingColumns,
    NullValues,
    UnsortedTimestamps,
)

# All 5 structural rules — used for registration / category checks.
ALL_STRUCTURAL_RULES = [
    DuplicateTimestamps,
    MissingColumns,
    InvalidDtypes,
    UnsortedTimestamps,
    NullValues,
]

# Rules that are still stubs — used to assert NotImplementedError is raised.
STUB_STRUCTURAL_RULES: list[type] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(df: pl.DataFrame) -> CanonicalDataset:
    """Wrap a DataFrame in a CanonicalDataset with a dummy source path.

    row_count is populated from len(df) so that rules using dataset.row_count
    for affected_rows (e.g. InvalidDtypes) behave correctly in tests.
    """
    return CanonicalDataset(df=df, source_path="test.csv", row_count=len(df))


def _make_context() -> RuleContext:
    """Return a default RuleContext (uses default ValidationConfig)."""
    return RuleContext()


# ---------------------------------------------------------------------------
# Shared structural-rule contract tests
# ---------------------------------------------------------------------------

class TestStructuralRulesRegistered:
    @pytest.mark.parametrize("rule_cls", ALL_STRUCTURAL_RULES)
    def test_rule_is_registered(self, rule_cls: type) -> None:
        """Every structural rule must be present in the global REGISTRY."""
        assert rule_cls in REGISTRY

    @pytest.mark.parametrize("rule_cls", ALL_STRUCTURAL_RULES)
    def test_rule_has_correct_category(self, rule_cls: type) -> None:
        """Every structural rule must declare Category.STRUCTURAL."""
        rule = rule_cls()
        assert rule.category == Category.STRUCTURAL

    @pytest.mark.parametrize("rule_cls", STUB_STRUCTURAL_RULES)
    def test_stub_rules_raise_not_implemented(self, rule_cls: type) -> None:
        """Rules not yet implemented must raise NotImplementedError."""
        rule = rule_cls()
        with pytest.raises(NotImplementedError):
            rule.validate(None, None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# MissingColumns — full test suite
# ---------------------------------------------------------------------------

class TestMissingColumns:
    """Tests for the MissingColumns validation rule."""

    # --- PASS cases ---------------------------------------------------------

    def test_pass_all_required_columns_present(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """All 6 required columns present → PASS."""
        result = MissingColumns().validate(_make_dataset(sample_ohlcv_df), _make_context())

        assert result.status == Status.PASS
        assert result.rule_id == "structural.missing_columns"
        assert result.severity == Severity.CRITICAL
        assert result.affected_rows == 0
        assert result.details == {}

    def test_pass_extra_columns_are_ignored(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """Extra columns beyond the 6 required must not cause a failure."""
        df = sample_ohlcv_df.with_columns(pl.lit("AAPL").alias("ticker"))
        result = MissingColumns().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS

    # --- FAIL cases ---------------------------------------------------------

    def test_fail_single_missing_column(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """Dropping one column → FAIL with exactly that column reported."""
        df = sample_ohlcv_df.drop("volume")
        result = MissingColumns().validate(_make_dataset(df), _make_context())

        assert result.status == Status.FAIL
        assert result.details["missing_columns"] == ["volume"]
        assert result.affected_rows == 0

    def test_fail_multiple_missing_columns(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """Dropping multiple columns → FAIL with all of them listed."""
        df = sample_ohlcv_df.drop(["low", "volume"])
        result = MissingColumns().validate(_make_dataset(df), _make_context())

        assert result.status == Status.FAIL
        assert set(result.details["missing_columns"]) == {"low", "volume"}
        assert result.affected_rows == 0

    def test_fail_all_columns_missing(self) -> None:
        """A completely empty schema (no columns) → FAIL listing all 6."""
        df = pl.DataFrame()
        result = MissingColumns().validate(_make_dataset(df), _make_context())

        assert result.status == Status.FAIL
        assert len(result.details["missing_columns"]) == 6
        assert result.affected_rows == 0

    # --- Message format checks ----------------------------------------------

    def test_fail_message_contains_count_and_names(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """Failure message must contain the count and the missing column names."""
        df = sample_ohlcv_df.drop(["low", "volume"])
        result = MissingColumns().validate(_make_dataset(df), _make_context())

        assert "2" in result.message
        assert "low" in result.message
        assert "volume" in result.message

    def test_pass_message_is_informative(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """Pass message must be non-empty and describe the outcome."""
        result = MissingColumns().validate(_make_dataset(sample_ohlcv_df), _make_context())

        assert result.message
        assert "present" in result.message.lower()


# ---------------------------------------------------------------------------
# InvalidDtypes — full test suite
# ---------------------------------------------------------------------------

class TestInvalidDtypes:
    """Tests for the InvalidDtypes validation rule."""

    # --- PASS cases ---------------------------------------------------------

    def test_pass_correct_dtypes(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """Standard sample DataFrame has correct dtypes → PASS."""
        result = InvalidDtypes().validate(_make_dataset(sample_ohlcv_df), _make_context())

        assert result.status == Status.PASS
        assert result.rule_id == "structural.invalid_dtypes"
        assert result.severity == Severity.CRITICAL
        assert result.details == {}

    def test_pass_extra_columns_are_ignored(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """Extra columns with any dtype must not cause a failure."""
        df = sample_ohlcv_df.with_columns(pl.lit("AAPL").alias("ticker"))
        result = InvalidDtypes().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS

    def test_pass_missing_column_not_reported(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """A missing column must be skipped, not treated as a dtype mismatch.

        MissingColumns owns absent columns; InvalidDtypes must not double-report.
        """
        df = sample_ohlcv_df.drop("volume")
        result = InvalidDtypes().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS
        assert "volume" not in str(result.details)

    # --- FAIL cases ---------------------------------------------------------

    def test_fail_single_wrong_dtype(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """One column with wrong dtype → FAIL with that column in details."""
        # Cast volume to Float64 to simulate a column that wasn't coerced correctly.
        df = sample_ohlcv_df.with_columns(pl.col("volume").cast(pl.Float64))
        result = InvalidDtypes().validate(_make_dataset(df), _make_context())

        assert result.status == Status.FAIL
        assert "volume" in result.details["mismatched_columns"]
        mismatch = result.details["mismatched_columns"]["volume"]
        assert mismatch["expected"] == "Int64"
        assert mismatch["actual"] == "Float64"

    def test_fail_multiple_wrong_dtypes(self) -> None:
        """Multiple wrong dtypes → FAIL with all mismatched columns in details."""
        df = pl.DataFrame({
            "timestamp": [datetime(2024, 1, 2, 9, 30)],
            "open":      ["100.0"],   # String instead of Float64
            "high":      [100.8],
            "low":       [99.5],
            "close":     [100.5],
            "volume":    [1000.0],    # Float64 instead of Int64
        })
        result = InvalidDtypes().validate(_make_dataset(df), _make_context())

        assert result.status == Status.FAIL
        mismatches = result.details["mismatched_columns"]
        assert "open" in mismatches
        assert "volume" in mismatches
        assert mismatches["volume"]["expected"] == "Int64"
        assert mismatches["volume"]["actual"] == "Float64"

    def test_fail_affected_rows_equals_row_count(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """affected_rows must equal the total dataset row count.

        Every row carries a wrong-typed value when a column dtype is wrong.
        """
        df = sample_ohlcv_df.with_columns(pl.col("volume").cast(pl.Float64))
        result = InvalidDtypes().validate(_make_dataset(df), _make_context())

        assert result.status == Status.FAIL
        assert result.affected_rows == len(sample_ohlcv_df)  # 10

    # --- Message format checks ----------------------------------------------

    def test_fail_message_contains_count_and_column_names(
        self, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        """Failure message must contain the mismatch count and column names."""
        df = sample_ohlcv_df.with_columns(pl.col("volume").cast(pl.Float64))
        result = InvalidDtypes().validate(_make_dataset(df), _make_context())

        assert "1" in result.message
        assert "volume" in result.message


# ---------------------------------------------------------------------------
# UnsortedTimestamps — full test suite
# ---------------------------------------------------------------------------

class TestUnsortedTimestamps:
    """Tests for the UnsortedTimestamps validation rule."""

    # --- PASS cases ---------------------------------------------------------

    def test_pass_already_sorted(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """The standard ascending-order fixture must PASS."""
        result = UnsortedTimestamps().validate(_make_dataset(sample_ohlcv_df), _make_context())

        assert result.status == Status.PASS
        assert result.rule_id == "structural.unsorted_timestamps"
        assert result.severity == Severity.WARNING
        assert result.affected_rows == 0
        assert result.details == {}

    def test_skip_empty_dataset(self) -> None:
        """Zero rows means nothing to compare -> PASS."""
        df = pl.DataFrame({"timestamp": []}, schema={"timestamp": pl.Datetime("us")})
        result = UnsortedTimestamps().validate(_make_dataset(df), _make_context())

        assert result.status == Status.SKIP

    def test_skip_single_row(self) -> None:
        """A single row has no predecessor to violate order against -> PASS."""
        df = pl.DataFrame({"timestamp": [datetime(2024, 1, 2, 9, 30)]})
        result = UnsortedTimestamps().validate(_make_dataset(df), _make_context())

        assert result.status == Status.SKIP

    def test_skip_timestamp_column_missing(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """No timestamp column -> PASS (skip); MissingColumns owns this concern."""
        df = sample_ohlcv_df.drop("timestamp")
        result = UnsortedTimestamps().validate(_make_dataset(df), _make_context())

        assert result.status == Status.SKIP

    def test_pass_duplicate_timestamps_not_flagged(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """Equal consecutive timestamps (ties) are DuplicateTimestamps' concern, not ours."""
        df = sample_ohlcv_df.with_columns(
            pl.when(pl.int_range(0, sample_ohlcv_df.height) == 1)
            .then(pl.col("timestamp").first())
            .otherwise(pl.col("timestamp"))
            .alias("timestamp")
        )
        result = UnsortedTimestamps().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS

    def test_pass_nulls_excluded_from_comparison(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """A null timestamp must not trigger a false violation on either side of it."""
        df = sample_ohlcv_df.with_columns(
            pl.when(pl.int_range(0, sample_ohlcv_df.height) == 3)
            .then(None)
            .otherwise(pl.col("timestamp"))
            .alias("timestamp")
        )
        result = UnsortedTimestamps().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS

    # --- WARN cases ----------------------------------------------------------

    def test_warn_single_out_of_order_row(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """Swapping two adjacent timestamps produces exactly one violation."""
        ts = sample_ohlcv_df["timestamp"].to_list()
        ts[3], ts[4] = ts[4], ts[3]
        df = sample_ohlcv_df.with_columns(pl.Series("timestamp", ts))

        result = UnsortedTimestamps().validate(_make_dataset(df), _make_context())

        assert result.status == Status.WARN
        assert result.severity == Severity.WARNING
        assert result.affected_rows == 1

    def test_warn_multiple_out_of_order_rows(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """Multiple independent swaps produce a matching violation count."""
        ts = sample_ohlcv_df["timestamp"].to_list()
        ts[2], ts[3] = ts[3], ts[2]
        ts[6], ts[7] = ts[7], ts[6]
        df = sample_ohlcv_df.with_columns(pl.Series("timestamp", ts))

        result = UnsortedTimestamps().validate(_make_dataset(df), _make_context())

        assert result.status == Status.WARN
        assert result.affected_rows == 2

    def test_warn_details_capped_at_max_rows_in_details(
        self, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        """details['violations'] is capped by config.max_rows_in_details,
        but affected_rows still reflects the full violation count."""
        # Reverse the whole series: every adjacent pair after the first becomes
        # a violation (9 violations across 10 rows).
        ts = list(reversed(sample_ohlcv_df["timestamp"].to_list()))
        df = sample_ohlcv_df.with_columns(pl.Series("timestamp", ts))

        context = RuleContext(config=ValidationConfig(max_rows_in_details=2))
        result = UnsortedTimestamps().validate(_make_dataset(df), context)

        assert result.status == Status.WARN
        assert result.affected_rows == 9
        assert len(result.details["violations"]) == 2

    def test_warn_violation_index_and_timestamps_correct(
        self, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        """The reported index/previous/current values must match the 0-based
        convention: index is the row that broke order (the later row)."""
        ts = sample_ohlcv_df["timestamp"].to_list()
        original_3, original_4 = ts[3], ts[4]
        ts[3], ts[4] = ts[4], ts[3]
        df = sample_ohlcv_df.with_columns(pl.Series("timestamp", ts))

        result = UnsortedTimestamps().validate(_make_dataset(df), _make_context())

        violation = result.details["violations"][0]
        assert violation["index"] == 4
        assert violation["previous_timestamp"] == original_4  # now at index 3
        assert violation["current_timestamp"] == original_3   # now at index 4

    # --- Message format checks ------------------------------------------------

    def test_warn_message_contains_count_and_first_violation(
        self, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        """Message must contain the violation count and the first violation's index."""
        ts = sample_ohlcv_df["timestamp"].to_list()
        ts[3], ts[4] = ts[4], ts[3]
        df = sample_ohlcv_df.with_columns(pl.Series("timestamp", ts))

        result = UnsortedTimestamps().validate(_make_dataset(df), _make_context())

        assert "1" in result.message
        assert "index 4" in result.message

    def test_pass_message_is_informative(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """Pass message must be non-empty and describe the outcome."""
        result = UnsortedTimestamps().validate(_make_dataset(sample_ohlcv_df), _make_context())

        assert result.message
        assert "sorted" in result.message.lower()


# ---------------------------------------------------------------------------
# DuplicateTimestamps — full test suite
# ---------------------------------------------------------------------------

class TestDuplicateTimestamps:
    """Tests for the DuplicateTimestamps validation rule."""

    # --- PASS cases ---------------------------------------------------------

    def test_pass_no_duplicates(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """The standard fixture has all-distinct timestamps -> PASS."""
        result = DuplicateTimestamps().validate(_make_dataset(sample_ohlcv_df), _make_context())

        assert result.status == Status.PASS
        assert result.rule_id == "structural.duplicate_timestamps"
        assert result.severity == Severity.CRITICAL
        assert result.affected_rows == 0
        assert result.details == {}

    def test_skip_empty_dataset(self) -> None:
        """Zero rows means nothing to compare -> PASS."""
        df = pl.DataFrame({"timestamp": []}, schema={"timestamp": pl.Datetime("us")})
        result = DuplicateTimestamps().validate(_make_dataset(df), _make_context())

        assert result.status == Status.SKIP

    def test_skip_single_row(self) -> None:
        """A single row can't duplicate anything -> PASS."""
        df = pl.DataFrame({"timestamp": [datetime(2024, 1, 2, 9, 30)]})
        result = DuplicateTimestamps().validate(_make_dataset(df), _make_context())

        assert result.status == Status.SKIP

    def test_skip_timestamp_column_missing(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """No timestamp column -> PASS (skip); MissingColumns owns this concern."""
        df = sample_ohlcv_df.drop("timestamp")
        result = DuplicateTimestamps().validate(_make_dataset(df), _make_context())

        assert result.status == Status.SKIP

    def test_pass_multiple_nulls_not_treated_as_duplicates(
        self, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        """Multiple null timestamps must not be flagged as duplicates of each
        other -- NullValues owns nulls, not this rule."""
        ts = sample_ohlcv_df["timestamp"].to_list()
        ts[2] = None
        ts[5] = None
        df = sample_ohlcv_df.with_columns(pl.Series("timestamp", ts))

        result = DuplicateTimestamps().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS

    # --- FAIL cases ----------------------------------------------------------

    def test_fail_single_duplicated_pair(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """Two rows sharing one timestamp value -> FAIL, affected_rows == 2."""
        ts = sample_ohlcv_df["timestamp"].to_list()
        ts[5] = ts[2]  # duplicate an earlier timestamp
        df = sample_ohlcv_df.with_columns(pl.Series("timestamp", ts))

        result = DuplicateTimestamps().validate(_make_dataset(df), _make_context())

        assert result.status == Status.FAIL
        assert result.severity == Severity.CRITICAL
        assert result.affected_rows == 2

    def test_fail_non_adjacent_duplicates_detected(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """Duplicates that are NOT adjacent in row order must still be caught.

        This is the key behavior locking in group_by/count over a
        shift(1)-based adjacency check: rows 0 and 9 share a value here but
        are far apart in row order.
        """
        ts = sample_ohlcv_df["timestamp"].to_list()
        ts[9] = ts[0]
        df = sample_ohlcv_df.with_columns(pl.Series("timestamp", ts))

        result = DuplicateTimestamps().validate(_make_dataset(df), _make_context())

        assert result.status == Status.FAIL
        assert result.affected_rows == 2
        reported_indices = {v["index"] for v in result.details["violations"]}
        assert reported_indices == {0, 9}

    def test_fail_unsorted_non_adjacent_duplicates_detected(self) -> None:
        """Duplicates must be found even when the dataset is not sorted --
        this rule must not depend on UnsortedTimestamps having run or passed.
        """
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 2, 9, 30),
                    datetime(2024, 1, 2, 9, 34),
                    datetime(2024, 1, 2, 9, 30),  # duplicates row 0, unsorted
                    datetime(2024, 1, 2, 9, 33),
                ]
            }
        )

        result = DuplicateTimestamps().validate(_make_dataset(df), _make_context())

        assert result.status == Status.FAIL
        assert result.affected_rows == 2
        reported_indices = {v["index"] for v in result.details["violations"]}
        assert reported_indices == {0, 2}

    def test_fail_three_way_duplicate(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """Three rows sharing one timestamp value -> affected_rows == 3, and
        each reported violation's count reflects the full group size."""
        ts = sample_ohlcv_df["timestamp"].to_list()
        ts[4] = ts[1]
        ts[7] = ts[1]
        df = sample_ohlcv_df.with_columns(pl.Series("timestamp", ts))

        result = DuplicateTimestamps().validate(_make_dataset(df), _make_context())

        assert result.status == Status.FAIL
        assert result.affected_rows == 3
        assert all(v["count"] == 3 for v in result.details["violations"])

    def test_fail_multiple_independent_duplicate_groups(
        self, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        """Two separate duplicated timestamp values -> affected_rows counts
        all rows across both groups (4 total from two pairs)."""
        ts = sample_ohlcv_df["timestamp"].to_list()
        ts[5] = ts[2]  # pair 1
        ts[8] = ts[6]  # pair 2
        df = sample_ohlcv_df.with_columns(pl.Series("timestamp", ts))

        result = DuplicateTimestamps().validate(_make_dataset(df), _make_context())

        assert result.status == Status.FAIL
        assert result.affected_rows == 4

    def test_fail_violations_sorted_by_original_index(
        self, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        """Reported violations must be in original row-index order, not
        grouped/shuffled order."""
        ts = sample_ohlcv_df["timestamp"].to_list()
        ts[8] = ts[1]
        df = sample_ohlcv_df.with_columns(pl.Series("timestamp", ts))

        result = DuplicateTimestamps().validate(_make_dataset(df), _make_context())

        indices = [v["index"] for v in result.details["violations"]]
        assert indices == sorted(indices)

    def test_fail_details_capped_at_max_rows_in_details(
        self, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        """details['violations'] is capped by config.max_rows_in_details,
        but affected_rows still reflects the full violation count."""
        ts = sample_ohlcv_df["timestamp"].to_list()
        ts[5] = ts[2]
        ts[8] = ts[6]
        df = sample_ohlcv_df.with_columns(pl.Series("timestamp", ts))

        context = RuleContext(config=ValidationConfig(max_rows_in_details=2))
        result = DuplicateTimestamps().validate(_make_dataset(df), context)

        assert result.status == Status.FAIL
        assert result.affected_rows == 4
        assert len(result.details["violations"]) == 2

    def test_fail_null_timestamps_excluded_but_real_duplicates_still_found(
        self, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        """A null timestamp present elsewhere in the data must not interfere
        with detecting a genuine duplicate among the non-null rows."""
        ts = sample_ohlcv_df["timestamp"].to_list()
        ts[3] = None
        ts[5] = ts[2]
        df = sample_ohlcv_df.with_columns(pl.Series("timestamp", ts))

        result = DuplicateTimestamps().validate(_make_dataset(df), _make_context())

        assert result.status == Status.FAIL
        assert result.affected_rows == 2
        reported_indices = {v["index"] for v in result.details["violations"]}
        assert reported_indices == {2, 5}

    # --- Message format checks ------------------------------------------------

    def test_fail_message_contains_row_count_and_first_timestamp(
        self, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        """Message must contain the affected row count and the first
        duplicated timestamp value."""
        ts = sample_ohlcv_df["timestamp"].to_list()
        original = ts[2]
        ts[5] = original
        df = sample_ohlcv_df.with_columns(pl.Series("timestamp", ts))

        result = DuplicateTimestamps().validate(_make_dataset(df), _make_context())

        assert "2" in result.message
        assert str(original) in result.message

    def test_pass_message_is_informative(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """Pass message must be non-empty and describe the outcome."""
        result = DuplicateTimestamps().validate(_make_dataset(sample_ohlcv_df), _make_context())

        assert result.message
        assert "duplicate" in result.message.lower()


# ---------------------------------------------------------------------------
# NullValues — full test suite
# ---------------------------------------------------------------------------

class TestNullValues:
    """Tests for the NullValues validation rule."""

    # --- PASS cases ---------------------------------------------------------

    def test_pass_no_nulls(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """The standard fixture has no nulls -> PASS."""
        result = NullValues().validate(_make_dataset(sample_ohlcv_df), _make_context())

        assert result.status == Status.PASS
        assert result.rule_id == "structural.null_values"
        assert result.severity == Severity.WARNING
        assert result.affected_rows == 0
        assert result.details == {}

    def test_pass_empty_dataset(self) -> None:
        """Zero rows means zero nulls by construction. This is a real verdict, not
        a skip: the columns exist and were inspected, so PASS is honest."""
        df = pl.DataFrame(
            {
                "timestamp": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
            },
            schema={
                "timestamp": pl.Datetime("us"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
            },
        )
        result = NullValues().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS

    def test_skip_all_required_columns_missing(self) -> None:
        """No required columns present -> PASS via the empty-checked-columns guard,
        not treated as 'everything is null'. MissingColumns owns this case."""
        df = pl.DataFrame({"ticker": ["AAPL", "AAPL"]})
        result = NullValues().validate(_make_dataset(df), _make_context())

        assert result.status == Status.SKIP
        assert result.affected_rows == 0
        assert result.details == {}

    def test_pass_extra_column_nulls_ignored(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """A null in a non-required extra column is out of scope -> PASS."""
        df = sample_ohlcv_df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("ticker"))
        result = NullValues().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS

    # --- WARN cases ----------------------------------------------------------

    def test_warn_single_column_single_null(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """One null in one column -> WARN, affected_rows == 1."""
        volume = sample_ohlcv_df["volume"].to_list()
        volume[2] = None
        df = sample_ohlcv_df.with_columns(pl.Series("volume", volume, dtype=pl.Int64))

        result = NullValues().validate(_make_dataset(df), _make_context())

        assert result.status == Status.WARN
        assert result.severity == Severity.WARNING
        assert result.affected_rows == 1
        assert result.details["null_counts"] == {"volume": 1}

    def test_warn_single_column_multiple_nulls(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """Multiple nulls in one column -> affected_rows equals that count."""
        volume = sample_ohlcv_df["volume"].to_list()
        volume[2] = None
        volume[5] = None
        volume[7] = None
        df = sample_ohlcv_df.with_columns(pl.Series("volume", volume, dtype=pl.Int64))

        result = NullValues().validate(_make_dataset(df), _make_context())

        assert result.status == Status.WARN
        assert result.affected_rows == 3
        assert result.details["null_counts"] == {"volume": 3}

    def test_warn_multiple_columns_same_row(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """Nulls in two columns on the SAME row -> affected_rows == 1, not 2.

        This is the key behavior locking in the any_horizontal (distinct-rows)
        design over a naive sum-of-per-column-counts approach.
        """
        volume = sample_ohlcv_df["volume"].to_list()
        close = sample_ohlcv_df["close"].to_list()
        volume[4] = None
        close[4] = None
        df = sample_ohlcv_df.with_columns(
            pl.Series("volume", volume, dtype=pl.Int64),
            pl.Series("close", close, dtype=pl.Float64),
        )

        result = NullValues().validate(_make_dataset(df), _make_context())

        assert result.status == Status.WARN
        assert result.affected_rows == 1
        assert result.details["null_counts"] == {"volume": 1, "close": 1}

    def test_warn_multiple_columns_different_rows(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """Nulls in two columns on DIFFERENT rows -> affected_rows == 2."""
        volume = sample_ohlcv_df["volume"].to_list()
        close = sample_ohlcv_df["close"].to_list()
        volume[2] = None
        close[6] = None
        df = sample_ohlcv_df.with_columns(
            pl.Series("volume", volume, dtype=pl.Int64),
            pl.Series("close", close, dtype=pl.Float64),
        )

        result = NullValues().validate(_make_dataset(df), _make_context())

        assert result.status == Status.WARN
        assert result.affected_rows == 2
        assert result.details["null_counts"] == {"volume": 1, "close": 1}

    def test_warn_null_counts_shape_and_filtering(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """details['null_counts'] must only include columns with count > 0."""
        volume = sample_ohlcv_df["volume"].to_list()
        volume[0] = None
        df = sample_ohlcv_df.with_columns(pl.Series("volume", volume, dtype=pl.Int64))

        result = NullValues().validate(_make_dataset(df), _make_context())

        assert set(result.details["null_counts"].keys()) == {"volume"}
        assert "open" not in result.details["null_counts"]

    def test_warn_null_timestamp_is_reported(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """A null in 'timestamp' itself is in scope for this rule.

        UnsortedTimestamps explicitly defers reporting null timestamps to
        NullValues; this test confirms that promise is kept.
        """
        ts = sample_ohlcv_df["timestamp"].to_list()
        ts[3] = None
        df = sample_ohlcv_df.with_columns(pl.Series("timestamp", ts))

        result = NullValues().validate(_make_dataset(df), _make_context())

        assert result.status == Status.WARN
        assert result.details["null_counts"]["timestamp"] == 1
        assert result.affected_rows == 1

    def test_warn_entire_column_null(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """A fully-null column (every row) must be handled without error,
        with null_counts and affected_rows reflecting the full row count."""
        df = sample_ohlcv_df.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("volume")
        )

        result = NullValues().validate(_make_dataset(df), _make_context())

        assert result.status == Status.WARN
        assert result.details["null_counts"]["volume"] == len(sample_ohlcv_df)
        assert result.affected_rows == len(sample_ohlcv_df)

    def test_warn_unparseable_source_value_reported_as_null(self) -> None:
        """Documents a known limitation: coerce_dtypes() casts numeric columns
        with strict=False, so an unparseable source value (e.g. a non-numeric
        string in a numeric column) is silently converted to null during
        ingestion. NullValues cannot distinguish that from a genuinely absent
        source value -- both look identical by the time this rule runs.
        See PROJECT_STATUS.md for further discussion.
        """
        from marketcheck.ingestion.schema import coerce_dtypes

        raw = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 2, 9, 30), datetime(2024, 1, 2, 9, 31)],
                "open": ["100.0", "N/A"],  # "N/A" cannot parse as Float64
                "high": [100.8, 101.0],
                "low": [99.5, 100.5],
                "close": [100.5, 100.8],
                "volume": [1000, 1100],
            }
        )
        df = coerce_dtypes(raw)
        result = NullValues().validate(_make_dataset(df), _make_context())

        assert result.status == Status.WARN
        assert result.details["null_counts"]["open"] == 1

    # --- Message format checks ------------------------------------------------

    def test_warn_message_contains_column_count_and_row_count(
        self, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        """Message must lead with column count, then row-level impact."""
        volume = sample_ohlcv_df["volume"].to_list()
        volume[2] = None
        df = sample_ohlcv_df.with_columns(pl.Series("volume", volume, dtype=pl.Int64))

        result = NullValues().validate(_make_dataset(df), _make_context())

        assert "1" in result.message
        assert "volume" in result.message

    def test_pass_message_is_informative(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """Pass message must be non-empty and describe the outcome."""
        result = NullValues().validate(_make_dataset(sample_ohlcv_df), _make_context())

        assert result.message
        assert "null" in result.message.lower()
