"""Tests for numerical validation rules."""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from marketcheck.models.config import ValidationConfig
from marketcheck.models.dataset import CanonicalDataset
from marketcheck.models.enums import Category, Severity, Status
from marketcheck.validators import REGISTRY
from marketcheck.validators.base import RuleContext
from marketcheck.validators.numerical import (
    ImpossibleValues,
    OhlcRangeViolation,
    SuspiciousPriceJump,
    VolumeAnomaly,
)

# All 4 numerical rules — used for registration / category checks.
ALL_NUMERICAL_RULES = [
    OhlcRangeViolation,
    VolumeAnomaly,
    SuspiciousPriceJump,
    ImpossibleValues,
]

# Rules that are still stubs — used to assert NotImplementedError is raised.
STUB_NUMERICAL_RULES = [
    SuspiciousPriceJump,
    ImpossibleValues,
]


def _volume_df(volumes: list[int | None]) -> pl.DataFrame:
    """Build a volume-only DataFrame (the only column VolumeAnomaly reads)."""
    return pl.DataFrame({"volume": volumes}, schema={"volume": pl.Int64})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(df: pl.DataFrame) -> CanonicalDataset:
    """Wrap a DataFrame in a CanonicalDataset with a dummy source path."""
    return CanonicalDataset(df=df, source_path="test.csv", row_count=len(df))


def _make_context() -> RuleContext:
    """Return a default RuleContext."""
    return RuleContext()


def _bar(
    open_: float = 100.0,
    high: float = 101.0,
    low: float = 99.0,
    close: float = 100.5,
) -> pl.DataFrame:
    """Build a single-row OHLCV DataFrame with overridable price values."""
    return pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 2, 9, 30)],
            "open": [open_],
            "high": [high],
            "low": [low],
            "close": [close],
            "volume": [1000],
        }
    )


# ---------------------------------------------------------------------------
# Shared numerical-rule contract tests
# ---------------------------------------------------------------------------

class TestNumericalRulesRegistered:
    @pytest.mark.parametrize("rule_cls", ALL_NUMERICAL_RULES)
    def test_rule_is_registered(self, rule_cls: type) -> None:
        """Every numerical rule must be present in the global REGISTRY."""
        assert rule_cls in REGISTRY

    @pytest.mark.parametrize("rule_cls", ALL_NUMERICAL_RULES)
    def test_rule_has_correct_category(self, rule_cls: type) -> None:
        """Every numerical rule must declare Category.NUMERICAL."""
        rule = rule_cls()
        assert rule.category == Category.NUMERICAL

    @pytest.mark.parametrize("rule_cls", STUB_NUMERICAL_RULES)
    def test_stub_rules_raise_not_implemented(self, rule_cls: type) -> None:
        """Rules not yet implemented must raise NotImplementedError."""
        rule = rule_cls()
        with pytest.raises(NotImplementedError):
            rule.validate(None, None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# OhlcRangeViolation — full test suite
# ---------------------------------------------------------------------------

class TestOhlcRangeViolation:
    """Tests for the OhlcRangeViolation validation rule."""

    # --- PASS cases ---------------------------------------------------------

    def test_pass_valid_fixture(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """The standard fixture has well-formed bars -> PASS."""
        result = OhlcRangeViolation().validate(_make_dataset(sample_ohlcv_df), _make_context())

        assert result.status == Status.PASS
        assert result.rule_id == "numerical.ohlc_range_violation"
        assert result.severity == Severity.CRITICAL
        assert result.affected_rows == 0
        assert result.details == {}

    def test_pass_empty_dataset(self) -> None:
        """Zero rows -> PASS."""
        df = _bar().head(0)
        result = OhlcRangeViolation().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS

    def test_pass_all_four_prices_equal(self) -> None:
        """A flat bar (open==high==low==close) is valid: all invariants use
        >= / <=, so equality must not be flagged."""
        df = _bar(open_=100.0, high=100.0, low=100.0, close=100.0)
        result = OhlcRangeViolation().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS

    def test_pass_open_and_close_at_boundaries(self) -> None:
        """open == low and close == high is valid (boundary equality)."""
        df = _bar(open_=99.0, high=101.0, low=99.0, close=101.0)
        result = OhlcRangeViolation().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS

    def test_pass_ohlc_column_missing(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """Dropping OHLC columns must not crash; MissingColumns owns absence.

        With only `close` left, no check has both operands present, so there is
        nothing comparable -> PASS/skip.
        """
        df = sample_ohlcv_df.drop(["open", "high", "low"])
        result = OhlcRangeViolation().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS

    def test_pass_nulls_excluded(self) -> None:
        """A null price must not produce a violation; NullValues owns nulls."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 2, 9, 30)],
                "open": [None],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000],
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
        result = OhlcRangeViolation().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS

    def test_pass_negative_prices_not_flagged_here(self) -> None:
        """Documents the deliberate scope boundary: a geometrically valid bar
        with negative prices PASSES this rule. Sign pathologies belong to
        numerical.impossible_values, not to this rule. See PROJECT_STATUS.md.
        """
        df = _bar(open_=-5.0, high=-1.0, low=-10.0, close=-3.0)
        result = OhlcRangeViolation().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS

    # --- FAIL cases ----------------------------------------------------------

    def test_fail_high_below_low(self) -> None:
        """Inverted range (high < low) -> FAIL, reported by name."""
        df = _bar(open_=100.0, high=98.0, low=99.0, close=100.0)
        result = OhlcRangeViolation().validate(_make_dataset(df), _make_context())

        assert result.status == Status.FAIL
        assert result.severity == Severity.CRITICAL
        assert result.affected_rows == 1
        assert "high < low" in result.details["violations"][0]["failed_checks"]

    def test_fail_close_above_high(self) -> None:
        """close > high -> FAIL, reported as 'high < close'."""
        df = _bar(open_=100.0, high=101.0, low=99.0, close=105.0)
        result = OhlcRangeViolation().validate(_make_dataset(df), _make_context())

        assert result.status == Status.FAIL
        assert "high < close" in result.details["violations"][0]["failed_checks"]

    def test_fail_open_above_high(self) -> None:
        """open > high -> FAIL, reported as 'high < open'."""
        df = _bar(open_=105.0, high=101.0, low=99.0, close=100.0)
        result = OhlcRangeViolation().validate(_make_dataset(df), _make_context())

        assert result.status == Status.FAIL
        assert "high < open" in result.details["violations"][0]["failed_checks"]

    def test_fail_open_below_low(self) -> None:
        """open < low -> FAIL, reported as 'low > open'."""
        df = _bar(open_=95.0, high=101.0, low=99.0, close=100.0)
        result = OhlcRangeViolation().validate(_make_dataset(df), _make_context())

        assert result.status == Status.FAIL
        assert "low > open" in result.details["violations"][0]["failed_checks"]

    def test_fail_close_below_low(self) -> None:
        """close < low -> FAIL, reported as 'low > close'."""
        df = _bar(open_=100.0, high=101.0, low=99.0, close=95.0)
        result = OhlcRangeViolation().validate(_make_dataset(df), _make_context())

        assert result.status == Status.FAIL
        assert "low > close" in result.details["violations"][0]["failed_checks"]

    def test_fail_multiple_checks_on_one_row(self) -> None:
        """A badly malformed bar can fail several checks at once, and all of
        them must be reported for that single row."""
        # high=90 is below low=99, below open=100, and below close=100.
        df = _bar(open_=100.0, high=90.0, low=99.0, close=100.0)
        result = OhlcRangeViolation().validate(_make_dataset(df), _make_context())

        failed = result.details["violations"][0]["failed_checks"]
        assert "high < low" in failed
        assert "high < open" in failed
        assert "high < close" in failed

    def test_fail_row_with_multiple_violations_counted_once(self) -> None:
        """affected_rows counts distinct ROWS, not individual failed checks.

        This locks in the distinct-row semantics (mirroring NullValues) over a
        naive sum-of-failed-checks count.
        """
        df = _bar(open_=100.0, high=90.0, low=99.0, close=100.0)
        result = OhlcRangeViolation().validate(_make_dataset(df), _make_context())

        assert len(result.details["violations"][0]["failed_checks"]) > 1
        assert result.affected_rows == 1

    def test_fail_multiple_bad_rows(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """Two malformed rows -> affected_rows == 2 with correct indices."""
        high = sample_ohlcv_df["high"].to_list()
        high[2] = 1.0   # far below low
        high[7] = 1.0
        df = sample_ohlcv_df.with_columns(pl.Series("high", high, dtype=pl.Float64))

        result = OhlcRangeViolation().validate(_make_dataset(df), _make_context())

        assert result.status == Status.FAIL
        assert result.affected_rows == 2
        assert {v["index"] for v in result.details["violations"]} == {2, 7}

    def test_fail_details_capped_at_max_rows_in_details(
        self, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        """details['violations'] is capped, but affected_rows is the full count."""
        df = sample_ohlcv_df.with_columns(pl.lit(1.0).alias("high"))

        context = RuleContext(config=ValidationConfig(max_rows_in_details=3))
        result = OhlcRangeViolation().validate(_make_dataset(df), context)

        assert result.status == Status.FAIL
        assert result.affected_rows == len(sample_ohlcv_df)
        assert len(result.details["violations"]) == 3

    def test_fail_violation_includes_price_values(self) -> None:
        """Each violation entry carries the row's OHLC values for diagnosis."""
        df = _bar(open_=100.0, high=98.0, low=99.0, close=100.0)
        result = OhlcRangeViolation().validate(_make_dataset(df), _make_context())

        violation = result.details["violations"][0]
        assert violation["open"] == 100.0
        assert violation["high"] == 98.0
        assert violation["low"] == 99.0
        assert violation["close"] == 100.0

    def test_fail_partial_columns_still_checked(self) -> None:
        """With only high and low present, the high<low check still applies.

        Mirrors NullValues' 'check what exists' approach rather than skipping
        the whole rule because open/close are absent.
        """
        df = pl.DataFrame({"high": [98.0], "low": [99.0]})
        result = OhlcRangeViolation().validate(_make_dataset(df), _make_context())

        assert result.status == Status.FAIL
        assert result.details["violations"][0]["failed_checks"] == ["high < low"]

    # --- Message format checks ------------------------------------------------

    def test_fail_message_contains_count_and_failed_check(self) -> None:
        """Message must contain the row count and name the first failed check."""
        df = _bar(open_=100.0, high=98.0, low=99.0, close=100.0)
        result = OhlcRangeViolation().validate(_make_dataset(df), _make_context())

        assert "1" in result.message
        assert "high < low" in result.message

    def test_pass_message_is_informative(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """Pass message must be non-empty and describe the outcome."""
        result = OhlcRangeViolation().validate(_make_dataset(sample_ohlcv_df), _make_context())

        assert result.message
        assert "ohlc" in result.message.lower()


# ---------------------------------------------------------------------------
# VolumeAnomaly — full test suite
#
# Defaults under test: multiplier=10.0, window=20, centered, high-side only.
# ---------------------------------------------------------------------------

class TestVolumeAnomaly:
    """Tests for the VolumeAnomaly validation rule."""

    # --- PASS cases ---------------------------------------------------------

    def test_pass_uniform_volume(self) -> None:
        """Perfectly flat volume has no outliers -> PASS."""
        result = VolumeAnomaly().validate(_make_dataset(_volume_df([1000] * 30)), _make_context())

        assert result.status == Status.PASS
        assert result.rule_id == "numerical.volume_anomaly"
        assert result.severity == Severity.WARNING
        assert result.affected_rows == 0
        assert result.details == {}

    def test_pass_normal_variation(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """The standard fixture's mild volume variation is not anomalous."""
        result = VolumeAnomaly().validate(_make_dataset(sample_ohlcv_df), _make_context())

        assert result.status == Status.PASS

    def test_pass_volume_column_missing(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """No volume column -> PASS (skip); MissingColumns owns absence."""
        df = sample_ohlcv_df.drop("volume")
        result = VolumeAnomaly().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS

    def test_pass_empty_dataset(self) -> None:
        """Zero rows -> PASS."""
        result = VolumeAnomaly().validate(_make_dataset(_volume_df([])), _make_context())

        assert result.status == Status.PASS

    def test_pass_genuine_event_sized_spike_not_flagged(self) -> None:
        """A 4x spike — the size of a real earnings/news day — must NOT be
        flagged at the default 10x threshold. This is the deliberate calibration:
        genuine market events run 2-5x, so flagging them would spam the report
        and destroy trust in it."""
        volumes = [1000] * 30
        volumes[15] = 4000
        result = VolumeAnomaly().validate(_make_dataset(_volume_df(volumes)), _make_context())

        assert result.status == Status.PASS

    def test_pass_all_zero_volume_not_flagged(self) -> None:
        """An entirely zero-volume stretch must not produce violations.

        Guards the zero-median trap: without it, `v > 10 * 0` would be true for
        any positive volume, and even here a naive implementation could misbehave.
        """
        result = VolumeAnomaly().validate(_make_dataset(_volume_df([0] * 30)), _make_context())

        assert result.status == Status.PASS

    def test_pass_zero_median_does_not_flag_positive_volume(self) -> None:
        """THE key trap: in a mostly-zero (illiquid/halted) stretch the rolling
        median is 0, so an unguarded `v > k * 0` would flag every positive bar.
        A lone small positive volume must NOT be reported."""
        volumes = [0] * 30
        volumes[15] = 5
        result = VolumeAnomaly().validate(_make_dataset(_volume_df(volumes)), _make_context())

        assert result.status == Status.PASS

    def test_pass_null_volume_excluded(self) -> None:
        """Null volume must not be flagged; NullValues owns nulls."""
        volumes: list[int | None] = [1000] * 30
        volumes[15] = None
        result = VolumeAnomaly().validate(_make_dataset(_volume_df(volumes)), _make_context())

        assert result.status == Status.PASS

    def test_pass_low_outlier_not_flagged_high_side_only(self) -> None:
        """Documents the deliberate v1 scope: this rule is HIGH-side only.
        A collapse from 1000 to 1 share is not reported, because low-side
        detection roughly doubles false positives on legitimately thin bars."""
        volumes = [1000] * 30
        volumes[15] = 1
        result = VolumeAnomaly().validate(_make_dataset(_volume_df(volumes)), _make_context())

        assert result.status == Status.PASS

    # --- WARN cases ----------------------------------------------------------

    def test_warn_single_large_spike(self) -> None:
        """A 100x spike — units-error territory — is flagged."""
        volumes = [1000] * 30
        volumes[15] = 100_000
        result = VolumeAnomaly().validate(_make_dataset(_volume_df(volumes)), _make_context())

        assert result.status == Status.WARN
        assert result.severity == Severity.WARNING
        assert result.affected_rows == 1
        v = result.details["violations"][0]
        assert v["index"] == 15
        assert v["volume"] == 100_000
        assert v["ratio"] == 100.0

    def test_warn_spike_does_not_mask_itself(self) -> None:
        """A single extreme outlier must still be caught — the specific failure
        mode that makes global z-score unusable here (the outlier inflates std
        and hides). A rolling MEDIAN baseline is unmoved by it."""
        volumes = [1000] * 30
        volumes[15] = 10_000_000
        result = VolumeAnomaly().validate(_make_dataset(_volume_df(volumes)), _make_context())

        assert result.status == Status.WARN
        assert result.details["violations"][0]["rolling_median"] == 1000.0

    def test_warn_multiple_spikes(self) -> None:
        """Several well-separated spikes are all reported, in index order."""
        volumes = [1000] * 40
        volumes[10] = 50_000
        volumes[30] = 80_000
        result = VolumeAnomaly().validate(_make_dataset(_volume_df(volumes)), _make_context())

        assert result.status == Status.WARN
        assert result.affected_rows == 2
        assert [v["index"] for v in result.details["violations"]] == [10, 30]

    def test_warn_spike_at_series_start_is_auditable(self) -> None:
        """Edge bars must NOT be blind spots: partial (min_samples=1) windows
        mean the very first bar is still checked."""
        volumes = [1000] * 30
        volumes[0] = 100_000
        result = VolumeAnomaly().validate(_make_dataset(_volume_df(volumes)), _make_context())

        assert result.status == Status.WARN
        assert result.details["violations"][0]["index"] == 0

    def test_warn_spike_at_series_end_is_auditable(self) -> None:
        """Same for the final bar."""
        volumes = [1000] * 30
        volumes[-1] = 100_000
        result = VolumeAnomaly().validate(_make_dataset(_volume_df(volumes)), _make_context())

        assert result.status == Status.WARN
        assert result.details["violations"][0]["index"] == 29

    def test_warn_custom_multiplier_is_honoured(self) -> None:
        """A 4x spike passes at the default 10x but is flagged at 3x, proving
        the threshold is genuinely configurable rather than hardcoded."""
        volumes = [1000] * 30
        volumes[15] = 4000

        assert (
            VolumeAnomaly().validate(_make_dataset(_volume_df(volumes)), _make_context()).status
            == Status.PASS
        )

        context = RuleContext(config=ValidationConfig(volume_anomaly_multiplier=3.0))
        result = VolumeAnomaly().validate(_make_dataset(_volume_df(volumes)), context)

        assert result.status == Status.WARN
        assert result.details["multiplier"] == 3.0

    def test_warn_custom_window_is_honoured(self) -> None:
        """The window size is configurable and echoed back in details."""
        volumes = [1000] * 30
        volumes[15] = 100_000
        context = RuleContext(config=ValidationConfig(volume_anomaly_window=5))
        result = VolumeAnomaly().validate(_make_dataset(_volume_df(volumes)), context)

        assert result.status == Status.WARN
        assert result.details["window"] == 5

    def test_warn_details_capped_at_max_rows_in_details(self) -> None:
        """details['violations'] is capped, affected_rows keeps the full count."""
        # Alternating baseline/spike keeps the rolling median at the low value
        # while producing many violations.
        volumes = [1000 if i % 2 == 0 else 100_000 for i in range(60)]
        context = RuleContext(config=ValidationConfig(max_rows_in_details=4))
        result = VolumeAnomaly().validate(_make_dataset(_volume_df(volumes)), context)

        assert result.status == Status.WARN
        assert result.affected_rows > 4
        assert len(result.details["violations"]) == 4

    def test_warn_violation_carries_diagnostic_fields(self) -> None:
        """Each violation reports volume, the baseline it was judged against, and
        the ratio — enough to assess it without reopening the source file."""
        volumes = [200] * 30
        volumes[10] = 5000
        result = VolumeAnomaly().validate(_make_dataset(_volume_df(volumes)), _make_context())

        v = result.details["violations"][0]
        assert v["volume"] == 5000
        assert v["rolling_median"] == 200.0
        assert v["ratio"] == 25.0

    # --- Message format checks ------------------------------------------------

    def test_pass_realistic_intraday_u_shape_no_false_positives(self) -> None:
        """THE decisive property of this rule's design, locked in as a regression.

        Real intraday volume is U-shaped: the open runs ~30x midday and the close
        ~15x. Across 3 sessions of 390 minute-bars, a centered rolling-median
        baseline must produce ZERO false positives, because each bar is judged
        against temporal neighbours that share its position in the profile.

        Measured alternatives on this exact series, for contrast:
          - global 10x-median  -> 132 bars flagged
          - global z-score > 3 ->  45 bars flagged
        Both would flag only legitimate open/close bars, making the report
        useless. This test fails if anyone swaps the baseline for a global
        statistic.
        """
        def session() -> list[int]:
            out: list[int] = []
            for minute in range(390):
                if minute < 15:
                    out.append(30_000)      # opening burst
                elif minute < 30:
                    out.append(15_000)
                elif minute > 375:
                    out.append(15_000)      # closing burst
                elif minute > 360:
                    out.append(8_000)
                else:
                    out.append(1_000)       # midday lull
            return out

        volumes = session() * 3
        result = VolumeAnomaly().validate(_make_dataset(_volume_df(volumes)), _make_context())

        assert result.status == Status.PASS
        assert result.affected_rows == 0

    def test_warn_real_spike_still_caught_inside_u_shape(self) -> None:
        """The U-shape tolerance must not blind the rule: a genuine 100x midday
        spike inside the same realistic profile is still flagged."""
        def session() -> list[int]:
            out: list[int] = []
            for minute in range(390):
                if minute < 30:
                    out.append(30_000)
                elif minute > 360:
                    out.append(15_000)
                else:
                    out.append(1_000)
            return out

        volumes = session() * 3
        volumes[200] = 100_000  # midday, 100x its neighbours
        result = VolumeAnomaly().validate(_make_dataset(_volume_df(volumes)), _make_context())

        assert result.status == Status.WARN
        assert 200 in [v["index"] for v in result.details["violations"]]

    # --- Message format checks (VolumeAnomaly) --------------------------------

    def test_warn_message_is_worded_as_review_not_corruption(self) -> None:
        """The message must invite review rather than assert corruption: a spike
        is indistinguishable from a genuine market event at this layer, and
        overclaiming would undermine trust in the whole report."""
        volumes = [1000] * 30
        volumes[15] = 100_000
        result = VolumeAnomaly().validate(_make_dataset(_volume_df(volumes)), _make_context())

        assert "review" in result.message.lower()
        for overclaim in ("corrupt", "invalid", "error"):
            assert overclaim not in result.message.lower()

    def test_warn_message_contains_count_threshold_and_example(self) -> None:
        """Message names the count, the threshold applied, and a concrete example."""
        volumes = [1000] * 30
        volumes[15] = 100_000
        result = VolumeAnomaly().validate(_make_dataset(_volume_df(volumes)), _make_context())

        assert "1 bar(s)" in result.message
        assert "10x" in result.message
        assert "index 15" in result.message

    def test_pass_message_is_informative(self) -> None:
        """Pass message must be non-empty and describe the outcome."""
        result = VolumeAnomaly().validate(_make_dataset(_volume_df([1000] * 30)), _make_context())

        assert result.message
        assert "volume" in result.message.lower()
