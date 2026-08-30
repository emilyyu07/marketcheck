"""Tests for structural validation rules."""

from __future__ import annotations

import polars as pl
import pytest

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
    InvalidDtypes,
    UnsortedTimestamps,
    NullValues,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(df: pl.DataFrame) -> CanonicalDataset:
    """Wrap a DataFrame in a CanonicalDataset with a dummy source path."""
    return CanonicalDataset(df=df, source_path="test.csv")


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
# Rule 1: MissingColumns — full test suite
# ---------------------------------------------------------------------------

class TestMissingColumns:
    """Tests for the MissingColumns validation rule."""

    # PASS cases — all required columns must be present, extra columns are ignored

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

    # FAIL cases — missing columns must be reported in details and message

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

    # Message format checks

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
