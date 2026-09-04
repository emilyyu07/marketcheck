"""Tests for temporal validation rules."""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from marketcheck.models.config import ValidationConfig
from marketcheck.models.dataset import CanonicalDataset
from marketcheck.models.enums import Category, Severity, Status
from marketcheck.validators import REGISTRY
from marketcheck.validators.base import RuleContext
from marketcheck.validators.temporal import (
    GapsWithinSession,
    MissingSessions,
    OutsideTradingHours,
    TimezoneInconsistency,
)

# All 4 temporal rules — used for registration / category checks.
ALL_TEMPORAL_RULES = [
    MissingSessions,
    GapsWithinSession,
    OutsideTradingHours,
    TimezoneInconsistency,
]

# Rules that are still stubs — used to assert NotImplementedError is raised.
STUB_TEMPORAL_RULES = [
    MissingSessions,
    GapsWithinSession,
    TimezoneInconsistency,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(df: pl.DataFrame) -> CanonicalDataset:
    """Wrap a DataFrame in a CanonicalDataset with a dummy source path."""
    return CanonicalDataset(df=df, source_path="test.csv", row_count=len(df))


def _make_context() -> RuleContext:
    """Return a default RuleContext (uses default ValidationConfig/MarketCalendar)."""
    return RuleContext()


# ---------------------------------------------------------------------------
# Shared temporal-rule contract tests
# ---------------------------------------------------------------------------

class TestTemporalRulesRegistered:
    @pytest.mark.parametrize("rule_cls", ALL_TEMPORAL_RULES)
    def test_rule_is_registered(self, rule_cls: type) -> None:
        """Every temporal rule must be present in the global REGISTRY."""
        assert rule_cls in REGISTRY

    @pytest.mark.parametrize("rule_cls", ALL_TEMPORAL_RULES)
    def test_rule_has_correct_category(self, rule_cls: type) -> None:
        """Every temporal rule must declare Category.TEMPORAL."""
        rule = rule_cls()
        assert rule.category == Category.TEMPORAL

    @pytest.mark.parametrize("rule_cls", STUB_TEMPORAL_RULES)
    def test_stub_rules_raise_not_implemented(self, rule_cls: type) -> None:
        """Rules not yet implemented must raise NotImplementedError."""
        rule = rule_cls()
        with pytest.raises(NotImplementedError):
            rule.validate(None, None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# OutsideTradingHours — full test suite
# ---------------------------------------------------------------------------

class TestOutsideTradingHours:
    """Tests for the OutsideTradingHours validation rule."""

    # --- PASS cases ---------------------------------------------------------

    def test_pass_all_within_regular_hours(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """The standard fixture (09:30-09:39) is entirely within regular hours -> PASS."""
        result = OutsideTradingHours().validate(_make_dataset(sample_ohlcv_df), _make_context())

        assert result.status == Status.PASS
        assert result.rule_id == "temporal.outside_trading_hours"
        assert result.severity == Severity.INFO
        assert result.affected_rows == 0
        assert result.details == {}

    def test_pass_empty_dataset(self) -> None:
        """Zero rows means nothing to check -> PASS."""
        df = pl.DataFrame({"timestamp": []}, schema={"timestamp": pl.Datetime("us")})
        result = OutsideTradingHours().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS

    def test_pass_timestamp_column_missing(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """No timestamp column -> PASS (skip); MissingColumns owns this concern."""
        df = sample_ohlcv_df.drop("timestamp")
        result = OutsideTradingHours().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS

    def test_pass_exact_open_boundary_is_regular_hours(self) -> None:
        """Exactly 09:30:00 is regular hours (market opens then) -> PASS."""
        df = pl.DataFrame({"timestamp": [datetime(2024, 1, 2, 9, 30, 0)]})
        result = OutsideTradingHours().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS

    def test_pass_one_second_before_close_is_regular_hours(self) -> None:
        """15:59:59 is still within regular hours -> PASS."""
        df = pl.DataFrame({"timestamp": [datetime(2024, 1, 2, 15, 59, 59)]})
        result = OutsideTradingHours().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS

    def test_pass_null_timestamps_excluded(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """A null timestamp must not be flagged; NullValues owns nulls, not this rule."""
        ts = sample_ohlcv_df["timestamp"].to_list()
        ts[3] = None
        df = sample_ohlcv_df.with_columns(pl.Series("timestamp", ts))

        result = OutsideTradingHours().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS

    def test_pass_weekend_date_not_flagged_time_of_day_only(self) -> None:
        """A timestamp on a non-trading day (e.g. a Saturday) at a regular-hours
        time-of-day is NOT flagged -- this rule checks time-of-day only, not
        whether the date itself is a valid session (that's MissingSessions'
        territory), so date validity must not affect this rule's outcome."""
        # 2024-01-06 is a Saturday.
        df = pl.DataFrame({"timestamp": [datetime(2024, 1, 6, 10, 0, 0)]})
        result = OutsideTradingHours().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS

    # --- WARN cases ----------------------------------------------------------

    def test_warn_exact_close_boundary_is_outside_hours(self) -> None:
        """Exactly 16:00:00 is NOT regular hours (market is closed at that
        instant) -> WARN. Locks in the half-open [09:30, 16:00) interval."""
        df = pl.DataFrame({"timestamp": [datetime(2024, 1, 2, 16, 0, 0)]})
        result = OutsideTradingHours().validate(_make_dataset(df), _make_context())

        assert result.status == Status.WARN
        assert result.severity == Severity.INFO
        assert result.affected_rows == 1

    def test_warn_before_open(self) -> None:
        """A pre-market timestamp (before 09:30) -> WARN."""
        df = pl.DataFrame({"timestamp": [datetime(2024, 1, 2, 8, 0, 0)]})
        result = OutsideTradingHours().validate(_make_dataset(df), _make_context())

        assert result.status == Status.WARN
        assert result.affected_rows == 1

    def test_warn_after_close(self) -> None:
        """An after-market timestamp (at/after 16:00) -> WARN."""
        df = pl.DataFrame({"timestamp": [datetime(2024, 1, 2, 20, 0, 0)]})
        result = OutsideTradingHours().validate(_make_dataset(df), _make_context())

        assert result.status == Status.WARN
        assert result.affected_rows == 1

    def test_warn_multiple_rows_outside_hours(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """Multiple rows outside regular hours -> affected_rows matches count."""
        ts = sample_ohlcv_df["timestamp"].to_list()
        ts[0] = datetime(2024, 1, 2, 8, 0, 0)   # pre-market
        ts[9] = datetime(2024, 1, 2, 17, 0, 0)  # after-hours
        df = sample_ohlcv_df.with_columns(pl.Series("timestamp", ts))

        result = OutsideTradingHours().validate(_make_dataset(df), _make_context())

        assert result.status == Status.WARN
        assert result.affected_rows == 2
        reported_indices = {v["index"] for v in result.details["violations"]}
        assert reported_indices == {0, 9}

    def test_warn_details_capped_at_max_rows_in_details(
        self, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        """details['violations'] is capped by config.max_rows_in_details, but
        affected_rows still reflects the full violation count."""
        ts = [datetime(2024, 1, 2, 20, 0, 0)] * len(sample_ohlcv_df)
        df = sample_ohlcv_df.with_columns(pl.Series("timestamp", ts))

        context = RuleContext(config=ValidationConfig(max_rows_in_details=2))
        result = OutsideTradingHours().validate(_make_dataset(df), context)

        assert result.status == Status.WARN
        assert result.affected_rows == len(sample_ohlcv_df)
        assert len(result.details["violations"]) == 2

    def test_warn_violation_details_shape(self) -> None:
        """Each violation entry must carry index, timestamp, and time_of_day."""
        ts_value = datetime(2024, 1, 2, 8, 0, 0)
        df = pl.DataFrame({"timestamp": [ts_value]})

        result = OutsideTradingHours().validate(_make_dataset(df), _make_context())

        violation = result.details["violations"][0]
        assert violation["index"] == 0
        assert violation["timestamp"] == ts_value
        assert violation["time_of_day"] == ts_value.time()

    # --- Message format checks ------------------------------------------------

    def test_warn_message_contains_count_and_hours_range(self) -> None:
        """Message must contain the affected row count and the regular-hours range."""
        df = pl.DataFrame({"timestamp": [datetime(2024, 1, 2, 8, 0, 0)]})
        result = OutsideTradingHours().validate(_make_dataset(df), _make_context())

        assert "1" in result.message
        assert "09:30" in result.message
        assert "16:00" in result.message

    def test_pass_message_is_informative(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """Pass message must be non-empty and describe the outcome."""
        result = OutsideTradingHours().validate(_make_dataset(sample_ohlcv_df), _make_context())

        assert result.message
        assert "trading hours" in result.message.lower()
