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
    GapsWithinSession,
    TimezoneInconsistency,
]


def _df_for_dates(date_strs: list[str], hour: int = 10, minute: int = 0) -> pl.DataFrame:
    """Build a timestamp-only DataFrame with one row per given date (YYYY-MM-DD).

    Times default to 10:00 (inside regular hours) so this fixture never
    incidentally trips OutsideTradingHours.
    """
    return pl.DataFrame(
        {
            "timestamp": [
                datetime(*(int(p) for p in d.split("-")), hour, minute)  # type: ignore[misc]
                for d in date_strs
            ]
        }
    )


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


# ---------------------------------------------------------------------------
# MissingSessions — full test suite
#
# Test dates are chosen against the REAL NYSE calendar, verified via
# MarketCalendar.valid_sessions():
#   - 2024-01-06/07 is a weekend (Sat/Sun)
#   - 2024-01-15 is MLK Day (holiday); 2024-01-12, 16, 17 are sessions
#   - 2024-11-28 is Thanksgiving (holiday); 2024-11-29 is a HALF session
#     (closes 13:00 ET)
# ---------------------------------------------------------------------------

class TestMissingSessions:
    """Tests for the MissingSessions validation rule."""

    # --- PASS cases ---------------------------------------------------------

    def test_pass_contiguous_sessions(self) -> None:
        """Four consecutive trading days, all present -> PASS."""
        df = _df_for_dates(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
        result = MissingSessions().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS
        assert result.rule_id == "temporal.missing_sessions"
        assert result.severity == Severity.WARNING
        assert result.affected_rows == 0
        assert result.details == {}

    def test_pass_weekend_gap_not_flagged(self) -> None:
        """Fri 2024-01-05 -> Mon 2024-01-08 skips a weekend, which is not a
        missing session because the exchange was closed."""
        df = _df_for_dates(["2024-01-05", "2024-01-08"])
        result = MissingSessions().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS

    def test_pass_holiday_gap_not_flagged(self) -> None:
        """2024-01-15 is MLK Day. Spanning it must not produce a violation."""
        df = _df_for_dates(["2024-01-12", "2024-01-16"])
        result = MissingSessions().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS

    def test_pass_half_session_counts_as_present(self) -> None:
        """2024-11-29 is a half session (closes 13:00 ET). Having any rows on it
        counts as present -- this rule judges date presence only, never coverage
        completeness within a session (that's GapsWithinSession's job).
        2024-11-28 (Thanksgiving) must also not be flagged."""
        df = _df_for_dates(["2024-11-27", "2024-11-29"])
        result = MissingSessions().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS

    def test_pass_single_date(self) -> None:
        """A single-day dataset spans a single-session range -> nothing missing."""
        df = _df_for_dates(["2024-01-02"])
        result = MissingSessions().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS

    def test_pass_timestamp_column_missing(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """No timestamp column -> PASS (skip); MissingColumns owns this."""
        df = sample_ohlcv_df.drop("timestamp")
        result = MissingSessions().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS

    def test_pass_empty_dataset(self) -> None:
        """Zero rows -> no derivable range -> PASS."""
        df = pl.DataFrame({"timestamp": []}, schema={"timestamp": pl.Datetime("us")})
        result = MissingSessions().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS

    def test_pass_all_null_timestamps(self) -> None:
        """All-null timestamps yield no usable dates -> PASS, not a crash.
        NullValues owns reporting the nulls themselves."""
        df = pl.DataFrame(
            {"timestamp": [None, None]}, schema={"timestamp": pl.Datetime("us")}
        )
        result = MissingSessions().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS

    def test_pass_intraday_rows_same_date_not_duplicated(self) -> None:
        """Many rows on one date collapse to a single present session."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 2, 9, 30),
                    datetime(2024, 1, 2, 12, 0),
                    datetime(2024, 1, 2, 15, 59),
                ]
            }
        )
        result = MissingSessions().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS

    # --- WARN cases ----------------------------------------------------------

    def test_warn_single_missing_session(self) -> None:
        """Omitting 2024-01-04 from an otherwise contiguous run -> 1 missing."""
        df = _df_for_dates(["2024-01-02", "2024-01-03", "2024-01-05"])
        result = MissingSessions().validate(_make_dataset(df), _make_context())

        assert result.status == Status.WARN
        assert result.severity == Severity.WARNING
        assert result.details["missing_sessions"] == ["2024-01-04"]
        assert result.details["missing_count"] == 1

    def test_warn_multiple_missing_sessions(self) -> None:
        """Omitting two mid-range sessions -> both reported, chronologically."""
        df = _df_for_dates(["2024-01-02", "2024-01-05", "2024-01-09"])
        result = MissingSessions().validate(_make_dataset(df), _make_context())

        assert result.status == Status.WARN
        # Expected sessions Jan 2-9: 2,3,4,5,8,9. Present: 2,5,9 -> missing 3,4,8.
        assert result.details["missing_sessions"] == ["2024-01-03", "2024-01-04", "2024-01-08"]
        assert result.details["missing_count"] == 3

    def test_warn_missing_session_adjacent_to_holiday(self) -> None:
        """A genuinely missing session next to a holiday is still caught: the
        holiday (MLK 01-15) is excluded from expectations, but 01-16 is not."""
        df = _df_for_dates(["2024-01-12", "2024-01-17"])
        result = MissingSessions().validate(_make_dataset(df), _make_context())

        assert result.status == Status.WARN
        assert result.details["missing_sessions"] == ["2024-01-16"]

    def test_warn_affected_rows_is_zero(self) -> None:
        """affected_rows stays 0 even when sessions are missing: absent sessions
        contribute no rows to point at. Same convention as MissingColumns."""
        df = _df_for_dates(["2024-01-02", "2024-01-05"])
        result = MissingSessions().validate(_make_dataset(df), _make_context())

        assert result.status == Status.WARN
        assert result.details["missing_count"] > 0
        assert result.affected_rows == 0

    def test_warn_details_reports_counts_and_range(self) -> None:
        """details carries expected/present counts and the derived range."""
        df = _df_for_dates(["2024-01-02", "2024-01-03", "2024-01-05"])
        result = MissingSessions().validate(_make_dataset(df), _make_context())

        d = result.details
        assert d["expected_session_count"] == 4      # Jan 2,3,4,5
        assert d["present_session_count"] == 3       # Jan 2,3,5
        assert d["range_start"] == "2024-01-02"
        assert d["range_end"] == "2024-01-05"

    def test_warn_details_capped_at_max_rows_in_details(self) -> None:
        """missing_sessions list is capped, but missing_count is the full total."""
        # Present only the first and last session of a ~2-week span.
        df = _df_for_dates(["2024-01-02", "2024-01-12"])
        context = RuleContext(config=ValidationConfig(max_rows_in_details=2))
        result = MissingSessions().validate(_make_dataset(df), context)

        assert result.status == Status.WARN
        assert result.details["missing_count"] == 7   # Jan 3,4,5,8,9,10,11
        assert len(result.details["missing_sessions"]) == 2

    def test_warn_nulls_do_not_collapse_range(self) -> None:
        """A null timestamp mixed in must not distort the derived date range or
        masquerade as a covered session."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 2, 10, 0),
                    None,
                    datetime(2024, 1, 5, 10, 0),
                ]
            },
            schema={"timestamp": pl.Datetime("us")},
        )
        result = MissingSessions().validate(_make_dataset(df), _make_context())

        assert result.status == Status.WARN
        assert result.details["range_start"] == "2024-01-02"
        assert result.details["range_end"] == "2024-01-05"
        assert result.details["missing_sessions"] == ["2024-01-03", "2024-01-04"]

    # --- Message format checks ------------------------------------------------

    def test_warn_message_contains_count_and_range(self) -> None:
        """Message must contain the missing count, the range, and a first example."""
        df = _df_for_dates(["2024-01-02", "2024-01-03", "2024-01-05"])
        result = MissingSessions().validate(_make_dataset(df), _make_context())

        assert "1" in result.message
        assert "2024-01-02" in result.message
        assert "2024-01-05" in result.message
        assert "2024-01-04" in result.message

    def test_pass_message_is_informative(self) -> None:
        """Pass message must be non-empty and describe the outcome."""
        df = _df_for_dates(["2024-01-02", "2024-01-03"])
        result = MissingSessions().validate(_make_dataset(df), _make_context())

        assert result.message
        assert "session" in result.message.lower()
