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
STUB_STRUCTURAL_RULES = [
    DuplicateTimestamps,
    NullValues,
]


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

    def test_pass_empty_dataset(self) -> None:
        """Zero rows means nothing to compare -> PASS."""
        df = pl.DataFrame({"timestamp": []}, schema={"timestamp": pl.Datetime("us")})
        result = UnsortedTimestamps().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS

    def test_pass_single_row(self) -> None:
        """A single row has no predecessor to violate order against -> PASS."""
        df = pl.DataFrame({"timestamp": [datetime(2024, 1, 2, 9, 30)]})
        result = UnsortedTimestamps().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS

    def test_pass_timestamp_column_missing(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """No timestamp column -> PASS (skip); MissingColumns owns this concern."""
        df = sample_ohlcv_df.drop("timestamp")
        result = UnsortedTimestamps().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS

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
