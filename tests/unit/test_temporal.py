"""Tests for temporal validation rules."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from marketcheck.models.config import ValidationConfig
from marketcheck.models.dataset import CanonicalDataset
from marketcheck.models.enums import Category, Severity, Status
from marketcheck.validators import REGISTRY
from marketcheck.validators.base import RuleContext
from marketcheck.validators.frequency import format_frequency, infer_frequency
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
# Every temporal rule is implemented; an empty list keeps the contract test
# in place for any future stub.
STUB_TEMPORAL_RULES: list[type] = []


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

    def test_skip_empty_dataset(self) -> None:
        """Zero rows means nothing to check -> PASS."""
        df = pl.DataFrame({"timestamp": []}, schema={"timestamp": pl.Datetime("us")})
        result = OutsideTradingHours().validate(_make_dataset(df), _make_context())

        assert result.status == Status.SKIP

    def test_skip_timestamp_column_missing(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """No timestamp column -> PASS (skip); MissingColumns owns this concern."""
        df = sample_ohlcv_df.drop("timestamp")
        result = OutsideTradingHours().validate(_make_dataset(df), _make_context())

        assert result.status == Status.SKIP

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

    def test_pass_documents_limit_when_gaps_outnumber_real_intervals(self) -> None:
        """HONEST LIMITATION of mode-based inference. In 09:30, 09:31, 09:32, then
        every second minute, the 2-minute *gap* spacing occurs more often than the
        true 1-minute spacing — so the mode concludes the grid is 2min and reports
        nothing. Mode inference assumes gaps are the minority; when they are not,
        the rule under-reports rather than guessing.
        """
        result = GapsWithinSession().validate(
            _make_dataset(_bars([0, 1, 2, 4, 6, 8, 10, 12])), _make_context()
        )

        assert result.status == Status.PASS
        assert result.details["inferred_frequency"] == "2min"

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

    def test_skip_timestamp_column_missing(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """No timestamp column -> PASS (skip); MissingColumns owns this."""
        df = sample_ohlcv_df.drop("timestamp")
        result = MissingSessions().validate(_make_dataset(df), _make_context())

        assert result.status == Status.SKIP

    def test_skip_empty_dataset(self) -> None:
        """Zero rows -> no derivable range -> PASS."""
        df = pl.DataFrame({"timestamp": []}, schema={"timestamp": pl.Datetime("us")})
        result = MissingSessions().validate(_make_dataset(df), _make_context())

        assert result.status == Status.SKIP

    def test_skip_all_null_timestamps(self) -> None:
        """All-null timestamps yield no usable dates, so there is no range to
        check -> SKIP, not a crash and not a vacuous pass. NullValues owns
        reporting the nulls themselves."""
        df = pl.DataFrame(
            {"timestamp": [None, None]}, schema={"timestamp": pl.Datetime("us")}
        )
        result = MissingSessions().validate(_make_dataset(df), _make_context())

        assert result.status == Status.SKIP

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


# ---------------------------------------------------------------------------
# GapsWithinSession
# ---------------------------------------------------------------------------

def _bars(
    minute_offsets: list[int],
    day: int = 2,
    start_hour: int = 9,
    start_minute: int = 30,
) -> pl.DataFrame:
    """Timestamps at the given minute offsets from 09:30 on 2024-01-{day}."""
    base = datetime(2024, 1, day, start_hour, start_minute)
    return pl.DataFrame(
        {"timestamp": [base + timedelta(minutes=m) for m in minute_offsets]},
        schema={"timestamp": pl.Datetime("us")},
    )


def _multi_day_bars(days: list[int], minute_offsets: list[int]) -> pl.DataFrame:
    """The same intraday offsets repeated across several sessions."""
    rows: list[datetime] = []
    for day in days:
        base = datetime(2024, 1, day, 9, 30)
        rows.extend(base + timedelta(minutes=m) for m in minute_offsets)
    return pl.DataFrame({"timestamp": rows}, schema={"timestamp": pl.Datetime("us")})


class TestGapsWithinSession:
    """Tests for the GapsWithinSession validation rule."""

    # --- PASS cases ---------------------------------------------------------

    def test_pass_contiguous_minute_bars(self) -> None:
        """A complete 1-minute run has no gaps."""
        result = GapsWithinSession().validate(
            _make_dataset(_bars(list(range(10)))), _make_context()
        )

        assert result.status == Status.PASS
        assert result.rule_id == "temporal.gaps_within_session"
        assert result.affected_rows == 0

    def test_pass_reports_inferred_frequency_on_clean_data(self) -> None:
        """Even on PASS the inferred grid is disclosed, so the reader can confirm
        the rule interpreted the data the way they expect."""
        result = GapsWithinSession().validate(
            _make_dataset(_bars(list(range(10)))), _make_context()
        )

        assert result.details["inferred_frequency"] == "1min"
        assert result.details["frequency_confidence"] == 1.0

    def test_pass_five_minute_grid(self) -> None:
        """The grid is inferred, not assumed to be 1-minute."""
        result = GapsWithinSession().validate(
            _make_dataset(_bars([0, 5, 10, 15, 20])), _make_context()
        )

        assert result.status == Status.PASS
        assert result.details["inferred_frequency"] == "5min"

    def test_skip_timestamp_column_missing(self) -> None:
        """MissingColumns owns absent columns."""
        df = pl.DataFrame({"close": [1.0, 2.0]}, schema={"close": pl.Float64})
        result = GapsWithinSession().validate(_make_dataset(df), _make_context())

        assert result.status == Status.SKIP

    def test_skip_single_row(self) -> None:
        """One row cannot form a delta."""
        result = GapsWithinSession().validate(_make_dataset(_bars([0])), _make_context())

        assert result.status == Status.SKIP

    def test_skip_empty_dataset(self) -> None:
        result = GapsWithinSession().validate(_make_dataset(_bars([])), _make_context())

        assert result.status == Status.SKIP

    def test_pass_standard_fixture(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """The shared fixture is 10 contiguous minute bars."""
        result = GapsWithinSession().validate(
            _make_dataset(sample_ohlcv_df), _make_context()
        )

        assert result.status == Status.PASS

    def test_skip_daily_data_self_skips(self) -> None:
        """KEY PROPERTY: for daily data every delta spans a session boundary, so
        no within-session interval exists to analyse. The rule skips with no
        special-casing — correct, because 'gap within a session' is meaningless
        when each session holds a single bar."""
        result = GapsWithinSession().validate(
            _make_dataset(_df_for_dates(["2024-01-02", "2024-01-03", "2024-01-04"])),
            _make_context(),
        )

        assert result.status == Status.SKIP
        assert "one bar per session" in result.message

    def test_pass_overnight_gap_not_reported(self) -> None:
        """The long gap between one session's close and the next session's open is
        not a within-session gap; MissingSessions owns absent days."""
        result = GapsWithinSession().validate(
            _make_dataset(_multi_day_bars([2, 3], [0, 1, 2, 3])), _make_context()
        )

        assert result.status == Status.PASS

    def test_skip_irregular_data_skipped(self) -> None:
        """THE DOMINANCE GUARD. Tick-like data has no fixed grid, so the modal
        delta is arbitrary and nearly every interval would look like a gap. The
        rule must decline rather than emit a flood of false positives."""
        result = GapsWithinSession().validate(
            _make_dataset(_bars([0, 1, 3, 6, 10, 15, 21])), _make_context()
        )

        assert result.status == Status.SKIP
        assert "could not infer" in result.message.lower()

    def test_skip_ambiguous_tie_skipped(self) -> None:
        """A perfect two-way tie (each spacing exactly half the intervals) is
        ambiguous. The dominance comparison is strict precisely so this is
        rejected rather than resolved arbitrarily by the tie-break."""
        # deltas: 1, 1, 5, 5
        result = GapsWithinSession().validate(
            _make_dataset(_bars([0, 1, 2, 7, 12])), _make_context()
        )

        assert result.status == Status.SKIP
        assert "could not infer" in result.message.lower()

    def test_skip_confidence_reported_when_inference_rejected(self) -> None:
        """The confidence is disclosed even on failure, so the skip is auditable."""
        result = GapsWithinSession().validate(
            _make_dataset(_bars([0, 1, 3, 6, 10, 15, 21])), _make_context()
        )

        assert "frequency_confidence" in result.details

    def test_pass_null_timestamps_ignored(self) -> None:
        """NullValues owns nulls; they cannot participate in a delta."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 2, 9, 30),
                    None,
                    datetime(2024, 1, 2, 9, 31),
                    datetime(2024, 1, 2, 9, 32),
                ]
            },
            schema={"timestamp": pl.Datetime("us")},
        )
        result = GapsWithinSession().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS

    def test_pass_duplicate_timestamps_do_not_create_gaps(self) -> None:
        """A zero delta is not a gap, and duplicates must not corrupt the modal
        spacing. DuplicateTimestamps owns that defect."""
        result = GapsWithinSession().validate(
            _make_dataset(_bars([0, 1, 1, 2, 3])), _make_context()
        )

        assert result.status == Status.PASS
        assert result.details["inferred_frequency"] == "1min"

    def test_pass_gap_below_configured_minimum(self) -> None:
        """Raising gap_min_missing_bars suppresses small holes."""
        df = _bars([0, 1, 2, 4, 5, 6])  # one missing bar at 09:33
        context = RuleContext(config=ValidationConfig(gap_min_missing_bars=2))
        result = GapsWithinSession().validate(_make_dataset(df), context)

        assert result.status == Status.PASS

    # --- WARN cases ---------------------------------------------------------

    def test_warn_single_gap_worked_example(self) -> None:
        """09:30-09:33 then 09:37: a 4-minute delta hiding 3 bars."""
        result = GapsWithinSession().validate(
            _make_dataset(_bars([0, 1, 2, 3, 7, 8])), _make_context()
        )

        assert result.status == Status.WARN
        assert result.severity == Severity.WARNING
        assert result.affected_rows == 1
        v = result.details["violations"][0]
        assert v["gap_start"] == datetime(2024, 1, 2, 9, 33)
        assert v["gap_end"] == datetime(2024, 1, 2, 9, 37)
        assert v["missing_bars"] == 3
        assert v["gap_duration"] == "4min"

    def test_warn_missing_bars_arithmetic_is_not_off_by_one(self) -> None:
        """REGRESSION GUARD. A 4-minute delta on a 1-minute grid hides 3 bars, not
        4, because gap_end itself is present. Reporting 4 would look plausible and
        be wrong."""
        result = GapsWithinSession().validate(
            _make_dataset(_bars([0, 4])), _make_context()
        )
        # Only one delta, so the grid cannot be inferred from this alone.
        assert result.status == Status.PASS

        # With a credible grid, the arithmetic must hold.
        result = GapsWithinSession().validate(
            _make_dataset(_bars([0, 1, 2, 6])), _make_context()
        )
        assert result.details["violations"][0]["missing_bars"] == 3

    def test_warn_single_missing_bar_reported_by_default(self) -> None:
        """Default gap_min_missing_bars=1 reports even one-bar holes: a backtest
        assuming every bar exists is affected by them."""
        result = GapsWithinSession().validate(
            _make_dataset(_bars([0, 1, 2, 4, 5, 6])), _make_context()
        )

        assert result.status == Status.WARN
        assert result.details["violations"][0]["missing_bars"] == 1

    def test_warn_multiple_gaps_counted_and_summarised(self) -> None:
        """Two holes: 1 bar missing, then 4 bars missing."""
        result = GapsWithinSession().validate(
            _make_dataset(_bars([0, 1, 2, 4, 5, 10, 11, 12])), _make_context()
        )

        assert result.status == Status.WARN
        assert result.details["gap_count"] == 2
        assert result.details["total_missing_bars"] == 5
        assert result.details["largest_gap_bars"] == 4
        assert result.affected_rows == 2

    def test_warn_largest_gap_distinguishes_scatter_from_outage(self) -> None:
        """largest_gap_bars is the key diagnostic: identical total loss means very
        different things depending on whether it is scattered or contiguous."""
        # 1-minute spacing must stay the clear majority, otherwise the gap
        # spacing itself becomes the inferred grid (see the limit test below).
        scattered = GapsWithinSession().validate(
            _make_dataset(_bars([0, 1, 2, 3, 4, 6, 7, 8, 9, 11])), _make_context()
        )
        contiguous = GapsWithinSession().validate(
            _make_dataset(_bars([0, 1, 2, 3, 4, 5, 6, 12])), _make_context()
        )

        assert scattered.details["largest_gap_bars"] == 1
        assert contiguous.details["largest_gap_bars"] == 5
        assert scattered.details["gap_count"] > contiguous.details["gap_count"]

    def test_warn_gap_detected_in_second_session(self) -> None:
        """Gaps are found per session, not only in the first one."""
        rows = [datetime(2024, 1, 2, 9, 30) + timedelta(minutes=m) for m in range(4)]
        rows += [datetime(2024, 1, 3, 9, 30) + timedelta(minutes=m) for m in [0, 1, 5, 6]]
        df = pl.DataFrame({"timestamp": rows}, schema={"timestamp": pl.Datetime("us")})
        result = GapsWithinSession().validate(_make_dataset(df), _make_context())

        assert result.status == Status.WARN
        assert result.details["violations"][0]["gap_start"] == datetime(2024, 1, 3, 9, 31)

    def test_warn_unsorted_input_still_detected(self) -> None:
        """Rules run independently and cannot assume sorted input, so detection
        runs on a sorted copy. UnsortedTimestamps owns row order."""
        ordered = _bars([0, 1, 2, 3, 7, 8])
        shuffled = ordered.sort("timestamp", descending=True)
        result = GapsWithinSession().validate(_make_dataset(shuffled), _make_context())

        assert result.status == Status.WARN
        assert result.details["violations"][0]["missing_bars"] == 3

    def test_warn_five_minute_grid_gap(self) -> None:
        """Gap arithmetic scales with the inferred grid, not a hardcoded minute."""
        result = GapsWithinSession().validate(
            _make_dataset(_bars([0, 5, 10, 30, 35])), _make_context()
        )

        assert result.status == Status.WARN
        assert result.details["inferred_frequency"] == "5min"
        assert result.details["violations"][0]["missing_bars"] == 3

    def test_warn_details_capped_but_totals_complete(self) -> None:
        """Totals are computed over the full gap set before details are capped."""
        # Six runs of five contiguous bars, each separated by a 3-minute step that
        # hides 2 bars: 24 one-minute deltas vs 5 gap deltas, so 1min stays modal.
        offsets: list[int] = []
        cursor = 0
        for _ in range(6):
            for _ in range(5):
                offsets.append(cursor)
                cursor += 1
            cursor += 2
        context = RuleContext(config=ValidationConfig(max_rows_in_details=2))
        result = GapsWithinSession().validate(_make_dataset(_bars(offsets)), context)

        assert len(result.details["violations"]) == 2
        assert result.details["gap_count"] > 2
        assert result.affected_rows == result.details["gap_count"]

    def test_warn_affected_rows_never_exceeds_row_count(self) -> None:
        """affected_rows counts EXISTING rows following a gap, not missing bars —
        which would otherwise exceed the dataset size on heavily-holed data."""
        df = _bars([0, 1, 2, 500])
        result = GapsWithinSession().validate(_make_dataset(df), _make_context())

        assert result.details["total_missing_bars"] > df.height
        assert result.affected_rows <= df.height

    # --- Message format checks -----------------------------------------------

    def test_warn_message_is_worded_as_review_not_corruption(self) -> None:
        """A genuine trading halt produces a legitimate gap and this rule cannot
        distinguish one from a truncated feed, so it must not claim corruption."""
        result = GapsWithinSession().validate(
            _make_dataset(_bars([0, 1, 2, 3, 7, 8])), _make_context()
        )

        lowered = result.message.lower()
        for overclaim in ("corrupt", "invalid"):
            assert overclaim not in lowered

    def test_warn_message_includes_counts_and_frequency(self) -> None:
        result = GapsWithinSession().validate(
            _make_dataset(_bars([0, 1, 2, 4, 5, 10, 11])), _make_context()
        )

        assert "2 gap(s)" in result.message
        assert "1min" in result.message

    def test_pass_message_is_informative(self) -> None:
        result = GapsWithinSession().validate(
            _make_dataset(_bars(list(range(5)))), _make_context()
        )

        assert "no gaps" in result.message.lower()


# ---------------------------------------------------------------------------
# Frequency inference helper (validators/frequency.py)
# ---------------------------------------------------------------------------

class TestInferFrequency:
    """Tests for the shared bar-frequency inference helper."""

    @staticmethod
    def _deltas(minutes: list[float]) -> pl.Series:
        return pl.Series("delta", [timedelta(minutes=m) for m in minutes])

    def test_mode_wins_over_mean_and_median(self) -> None:
        """The mode is used because gaps skew the mean and can defeat the median."""
        freq, confidence = infer_frequency(self._deltas([1, 1, 1, 1, 60]), 0.5)

        assert freq == timedelta(minutes=1)
        assert confidence == 0.8

    def test_empty_deltas_uninferable(self) -> None:
        """The daily-data path: no within-session intervals exist."""
        freq, confidence = infer_frequency(
            pl.Series("delta", [], dtype=pl.Duration("us")), 0.5
        )

        assert freq is None
        assert confidence == 0.0

    def test_irregular_deltas_rejected(self) -> None:
        freq, _ = infer_frequency(self._deltas([1, 2, 3, 4, 5]), 0.5)

        assert freq is None

    def test_perfect_tie_rejected_as_ambiguous(self) -> None:
        """Strict majority: at a 50/50 tie neither candidate is credible."""
        freq, confidence = infer_frequency(self._deltas([1, 1, 5, 5]), 0.5)

        assert freq is None
        assert confidence == 0.5

    def test_tie_break_is_deterministic(self) -> None:
        """polars' Series.mode() may return several values in non-deterministic
        order, which would make the rule answer differently on identical input.
        The helper sorts explicitly, so repeated calls must agree."""
        deltas = self._deltas([1, 1, 5, 5, 1])
        results = {infer_frequency(deltas, 0.5)[0] for _ in range(20)}

        assert len(results) == 1

    def test_dominance_is_configurable(self) -> None:
        deltas = self._deltas([1, 1, 1, 9, 9])  # 1min at 60%

        assert infer_frequency(deltas, 0.5)[0] == timedelta(minutes=1)
        assert infer_frequency(deltas, 0.9)[0] is None

    def test_nulls_ignored(self) -> None:
        deltas = pl.Series("delta", [timedelta(minutes=1), None, timedelta(minutes=1)])
        freq, confidence = infer_frequency(deltas, 0.5)

        assert freq == timedelta(minutes=1)
        assert confidence == 1.0

    def test_non_positive_modal_delta_rejected(self) -> None:
        """Zero deltas would mean division by zero downstream."""
        freq, _ = infer_frequency(self._deltas([0, 0, 0, 1]), 0.5)

        assert freq is None


class TestFormatFrequency:
    """Tests for the human-readable duration formatter."""

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (60, "1min"),
            (300, "5min"),
            (900, "15min"),
            (3600, "1h"),
            (14400, "4h"),
            (86400, "1d"),
            (30, "30s"),
            (90, "1min30s"),
        ],
    )
    def test_formats_common_spacings(self, seconds: int, expected: str) -> None:
        assert format_frequency(timedelta(seconds=seconds)) == expected

    def test_zero_duration(self) -> None:
        assert format_frequency(timedelta(0)) == "0s"

    def test_output_is_json_safe_string(self) -> None:
        """Durations are pre-formatted because a raw timedelta serialises to
        ISO-8601 ('PT4M') in JSON output, which is valid but unreadable."""
        import json

        value = format_frequency(timedelta(minutes=4))
        assert json.dumps({"d": value}) == '{"d": "4min"}'


# ---------------------------------------------------------------------------
# TimezoneInconsistency
# ---------------------------------------------------------------------------

def _tz_dataset(
    timestamps: list[datetime], source_timezone: str | None = "naive"
) -> CanonicalDataset:
    """Wrap timestamps in a CanonicalDataset carrying timezone provenance."""
    df = pl.DataFrame(
        {"timestamp": timestamps}, schema={"timestamp": pl.Datetime("us")}
    )
    return CanonicalDataset(
        df=df,
        source_path="test.csv",
        row_count=len(timestamps),
        source_timezone=source_timezone,
    )


def _session(start_hour: int, start_minute: int = 30, bars: int = 390) -> list[datetime]:
    """`bars` one-minute timestamps beginning at the given time on 2024-01-02.

    390 bars is a full NYSE session, so the default spans exactly 6.5 hours.
    """
    base = datetime(2024, 1, 2, start_hour, start_minute)
    return [base + timedelta(minutes=i) for i in range(bars)]


class TestTimezoneInconsistency:
    """Tests for the TimezoneInconsistency validation rule."""

    # --- PASS cases ---------------------------------------------------------

    def test_pass_correct_et_session(self) -> None:
        """A full 09:30-15:59 ET session is exactly what every rule assumes."""
        result = TimezoneInconsistency().validate(
            _tz_dataset(_session(9)), _make_context()
        )

        assert result.status == Status.PASS
        assert result.rule_id == "temporal.timezone_inconsistency"
        assert result.affected_rows == 0

    def test_pass_declared_timezone_is_not_an_inconsistency(self) -> None:
        """A source that declared a timezone consistently is NOT flagged: ingestion
        converts it to ET wall-clock, so reading it was correct. Flagging it would
        be a false positive on good data."""
        result = TimezoneInconsistency().validate(
            _tz_dataset(_session(9), source_timezone="UTC"), _make_context()
        )

        assert result.status == Status.PASS

    def test_pass_extended_hours_data(self) -> None:
        """04:00-20:00 spans 16 hours against a 6.5-hour session, so NO shift can
        align it. Safe without special-casing — the alignment bar simply cannot be
        met. OutsideTradingHours owns reporting these bars."""
        result = TimezoneInconsistency().validate(
            _tz_dataset(_session(4, 0, bars=960)), _make_context()
        )

        assert result.status == Status.PASS
        assert "no uniform timezone shift" in result.message.lower()

    def test_pass_partially_aligned_data_not_blamed_on_timezone(self) -> None:
        """Data already mostly inside the session needs no explanation."""
        result = TimezoneInconsistency().validate(
            _tz_dataset(_session(10, 0, bars=300)), _make_context()
        )

        assert result.status == Status.PASS
        assert "consistent with ET" in result.message

    # --- SKIP cases ---------------------------------------------------------

    def test_skip_timestamp_column_missing(self) -> None:
        """MissingColumns owns absent columns."""
        df = pl.DataFrame({"close": [1.0, 2.0]}, schema={"close": pl.Float64})
        result = TimezoneInconsistency().validate(_make_dataset(df), _make_context())

        assert result.status == Status.SKIP

    def test_skip_all_null_timestamps(self) -> None:
        """NullValues owns nulls; nothing survives to infer from."""
        df = pl.DataFrame(
            {"timestamp": [None, None]}, schema={"timestamp": pl.Datetime("us")}
        )
        result = TimezoneInconsistency().validate(_make_dataset(df), _make_context())

        assert result.status == Status.SKIP

    def test_skip_daily_data_stamped_at_midnight(self) -> None:
        """THE TRAP THIS RULE MUST NOT FALL INTO. Daily bars are commonly stamped
        00:00, all outside the session, and a +10:00 shift would move them inside.
        Without the distinct-times guard every daily dataset would be reported as
        timezone-shifted at CRITICAL severity."""
        daily = [datetime(2024, 1, d, 0, 0) for d in (2, 3, 4, 5, 8, 9, 10, 11, 12, 15)]
        result = TimezoneInconsistency().validate(_tz_dataset(daily), _make_context())

        assert result.status == Status.SKIP
        assert "distinct time" in result.message

    def test_skip_daily_data_stamped_at_close(self) -> None:
        """Same guard, with daily bars stamped 16:00 instead of midnight."""
        daily = [datetime(2024, 1, d, 16, 0) for d in (2, 3, 4, 5, 8, 9, 10, 11)]
        result = TimezoneInconsistency().validate(_tz_dataset(daily), _make_context())

        assert result.status == Status.SKIP

    def test_skip_too_few_distinct_times_reports_the_count(self) -> None:
        result = TimezoneInconsistency().validate(
            _tz_dataset([datetime(2024, 1, 2, 0, 0)]), _make_context()
        )

        assert result.details["distinct_times_of_day"] == 1

    # --- FAIL: metadata path -------------------------------------------------

    def test_fail_mixed_aware_and_naive_source(self) -> None:
        """The only unambiguously CRITICAL metadata case: rows that do not share a
        frame of reference. Previously invisible — the unparseable rows became
        nulls and only NullValues reported them, with no hint of the cause."""
        result = TimezoneInconsistency().validate(
            _tz_dataset(_session(9), source_timezone="mixed"), _make_context()
        )

        assert result.status == Status.FAIL
        assert result.severity == Severity.CRITICAL
        assert "mix" in result.message.lower()
        assert "null" in result.message.lower()

    def test_fail_mixed_source_reported_even_when_values_align(self) -> None:
        """Mixed representations are an inconsistency regardless of how the
        surviving values happen to look."""
        result = TimezoneInconsistency().validate(
            _tz_dataset(_session(9), source_timezone="mixed"), _make_context()
        )

        assert result.status == Status.FAIL
        assert result.details["source_timezone"] == "mixed"

    # --- FAIL: heuristic path ------------------------------------------------

    def test_fail_utc_naive_session_detected(self) -> None:
        """The common real-world case: a vendor ships UTC with no offset, so there
        is no metadata to inspect. 09:30 ET becomes 14:30 naive."""
        result = TimezoneInconsistency().validate(
            _tz_dataset(_session(14)), _make_context()
        )

        assert result.status == Status.FAIL
        assert result.details["suggested_shift"] == "-5:00"
        assert result.details["suggested_shift_minutes"] == -300

    def test_fail_reports_alignment_evidence(self) -> None:
        """A CRITICAL finding must show its working."""
        result = TimezoneInconsistency().validate(
            _tz_dataset(_session(14)), _make_context()
        )

        assert result.details["current_alignment"] < 0.5
        assert result.details["alignment_after_shift"] >= 0.9

    def test_fail_half_hour_offset_detected(self) -> None:
        """Half-hour zones exist and matter: a provider stamping India-local time
        (+05:30) appears shifted from ET by -10:30. Whole-hour-only search would
        miss this."""
        result = TimezoneInconsistency().validate(
            _tz_dataset(_session(20, 0)), _make_context()
        )

        assert result.status == Status.FAIL
        assert result.details["suggested_shift"].endswith(":30")

    def test_fail_affected_rows_is_every_row(self) -> None:
        """A wrong timezone is a property of the column, not of individual bars, so
        every row is affected — unlike every other rule, which counts offenders."""
        timestamps = _session(14)
        result = TimezoneInconsistency().validate(
            _tz_dataset(timestamps), _make_context()
        )

        assert result.affected_rows == len(timestamps)

    def test_fail_is_deterministic(self) -> None:
        """Equal-scoring shifts must not resolve by iteration order. Repeated runs
        on identical input must agree — the same class of defect as polars'
        non-deterministic mode() tie ordering."""
        timestamps = _session(14)
        shifts = {
            TimezoneInconsistency()
            .validate(_tz_dataset(timestamps), _make_context())
            .details["suggested_shift"]
            for _ in range(10)
        }

        assert len(shifts) == 1

    # --- Calibration is auditable --------------------------------------------

    def test_calibration_constants_echoed_on_fail(self) -> None:
        """The thresholds are not configurable, but hiding them would make a
        CRITICAL finding unverifiable."""
        result = TimezoneInconsistency().validate(
            _tz_dataset(_session(14)), _make_context()
        )

        assert result.details["min_distinct_times_of_day"] == 10
        assert result.details["min_alignment_after_shift"] == 0.90
        assert result.details["max_current_alignment"] == 0.50

    def test_calibration_constants_echoed_on_pass(self) -> None:
        result = TimezoneInconsistency().validate(
            _tz_dataset(_session(9)), _make_context()
        )

        assert "min_distinct_times_of_day" in result.details

    # --- Message wording ------------------------------------------------------

    def test_fail_message_states_the_consequence(self) -> None:
        """CRITICAL is justified by scope: this does not invalidate one finding, it
        invalidates every temporal rule. The message must say so."""
        result = TimezoneInconsistency().validate(
            _tz_dataset(_session(14)), _make_context()
        )

        lowered = result.message.lower()
        assert "unreliable" in lowered
        assert "et" in lowered

    def test_fail_message_names_the_suggested_shift(self) -> None:
        result = TimezoneInconsistency().validate(
            _tz_dataset(_session(14)), _make_context()
        )

        assert "-5:00" in result.message


class TestTimezoneAndTradingHoursTogether:
    """DD19: both rules may fire. This one names the cause, the other the symptom."""

    def test_both_rules_report_a_shifted_session(self) -> None:
        dataset = _tz_dataset(_session(14))
        tz_result = TimezoneInconsistency().validate(dataset, _make_context())
        hours_result = OutsideTradingHours().validate(dataset, _make_context())

        assert tz_result.status == Status.FAIL
        assert hours_result.status == Status.WARN

    def test_timezone_rule_is_the_more_severe_of_the_two(self) -> None:
        """The cause outranks the symptom: a wrong timezone makes the data unusable,
        while bars outside hours are merely noteworthy."""
        dataset = _tz_dataset(_session(14))

        assert (
            TimezoneInconsistency().validate(dataset, _make_context()).severity
            == Severity.CRITICAL
        )
        assert (
            OutsideTradingHours().validate(dataset, _make_context()).severity
            == Severity.INFO
        )
