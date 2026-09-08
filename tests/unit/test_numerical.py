"""Tests for numerical validation rules."""

from __future__ import annotations

from datetime import datetime, timedelta

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
STUB_NUMERICAL_RULES: list[type] = []


def _price_df(
    closes: list[float | None], dates: list[str] | None = None
) -> pl.DataFrame:
    """Build a close/timestamp DataFrame for price-jump testing.

    Args:
        closes: close prices, one per row.
        dates: optional YYYY-MM-DD per row. Defaults to all rows on one date
            (i.e. every transition is intraday). Times increment by a minute so
            ordering is well defined.
    """
    if dates is None:
        dates = ["2024-01-02"] * len(closes)
    stamps = []
    per_date_counter: dict[str, int] = {}
    for d in dates:
        idx = per_date_counter.get(d, 0)
        per_date_counter[d] = idx + 1
        y, m, day = (int(p) for p in d.split("-"))
        stamps.append(datetime(y, m, day, 10, 0) + timedelta(minutes=idx))
    return pl.DataFrame(
        {"timestamp": stamps, "close": closes},
        schema={"timestamp": pl.Datetime("us"), "close": pl.Float64},
    )


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


# ---------------------------------------------------------------------------
# ImpossibleValues — full test suite (rule 14)
#
# Asymmetry under test: prices flagged at <= 0, volume only at < 0
# (volume == 0 is legitimate). Plus NaN price detection, and no-trade-bar
# tagging for the vendor 0/0/0/0 + volume 0 encoding.
# ---------------------------------------------------------------------------

class TestImpossibleValues:
    """Tests for the ImpossibleValues validation rule."""

    # --- PASS cases ---------------------------------------------------------

    def test_pass_valid_fixture(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """The standard fixture has no impossible values -> PASS."""
        result = ImpossibleValues().validate(_make_dataset(sample_ohlcv_df), _make_context())

        assert result.status == Status.PASS
        assert result.rule_id == "numerical.impossible_values"
        assert result.severity == Severity.CRITICAL
        assert result.affected_rows == 0
        assert result.details == {}

    def test_pass_empty_dataset(self) -> None:
        """Zero rows -> PASS."""
        result = ImpossibleValues().validate(_make_dataset(_bar().head(0)), _make_context())

        assert result.status == Status.PASS

    def test_pass_zero_volume_is_legitimate(self) -> None:
        """THE key asymmetry: volume == 0 is legitimate (illiquid name, halted
        trading, a session with no prints) and must NOT be flagged, even though
        a price of 0 would be."""
        df = _bar().with_columns(pl.lit(0, dtype=pl.Int64).alias("volume"))
        result = ImpossibleValues().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS

    def test_pass_all_columns_missing(self) -> None:
        """No price or volume columns -> nothing checkable -> PASS."""
        df = pl.DataFrame({"timestamp": [datetime(2024, 1, 2, 9, 30)]})
        result = ImpossibleValues().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS

    def test_pass_null_prices_excluded(self) -> None:
        """A null price is NullValues' concern, not an impossible value."""
        df = pl.DataFrame(
            {
                "open": [None],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": [10],
            },
            schema={
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
            },
        )
        result = ImpossibleValues().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS

    # --- FAIL cases: non-positive prices -------------------------------------

    def test_fail_negative_price(self) -> None:
        """A negative close is impossible -> FAIL."""
        df = _bar(close=-3.0)
        result = ImpossibleValues().validate(_make_dataset(df), _make_context())

        assert result.status == Status.FAIL
        assert result.severity == Severity.CRITICAL
        assert result.affected_rows == 1
        assert "close <= 0" in result.details["violations"][0]["failed_checks"]

    def test_fail_zero_price_with_positive_volume(self) -> None:
        """A zero price is never a real trade -> FAIL (and with positive volume
        it is NOT the vendor no-trade pattern)."""
        df = _bar(low=0.0)
        result = ImpossibleValues().validate(_make_dataset(df), _make_context())

        assert result.status == Status.FAIL
        assert "low <= 0" in result.details["violations"][0]["failed_checks"]

    def test_fail_geometrically_valid_negative_bar_is_caught_here(self) -> None:
        """Closes the gap OhlcRangeViolation deliberately left open: this bar is
        geometrically self-consistent (and PASSES that rule) but is
        economically impossible, so THIS rule must catch it."""
        df = _bar(open_=-5.0, high=-1.0, low=-10.0, close=-3.0)

        assert OhlcRangeViolation().validate(_make_dataset(df), _make_context()).status == (
            Status.PASS
        )

        result = ImpossibleValues().validate(_make_dataset(df), _make_context())
        assert result.status == Status.FAIL
        failed = result.details["violations"][0]["failed_checks"]
        assert set(failed) == {"open <= 0", "high <= 0", "low <= 0", "close <= 0"}

    # --- FAIL cases: NaN ------------------------------------------------------

    def test_fail_nan_price_detected(self) -> None:
        """NaN is flagged. This is the hole that spanned the whole ruleset:
        is_null() is False for NaN, the dtype is still Float64, and every
        comparison against NaN is False, so no other rule catches it."""
        df = _bar(open_=float("nan"))
        result = ImpossibleValues().validate(_make_dataset(df), _make_context())

        assert result.status == Status.FAIL
        assert "open is NaN" in result.details["violations"][0]["failed_checks"]

    def test_fail_nan_in_high_evades_every_other_rule(self) -> None:
        """The decisive case for having an explicit is_nan() check.

        Polars uses a TOTAL ORDER where NaN sorts above all numbers (not
        IEEE-754). So a NaN in `high` makes every "high < ..." comparison False
        and `OhlcRangeViolation` passes it, while `NullValues` also passes it
        (is_null() is False for NaN). Without this rule, a NaN in `high` is
        undetectable by anything in the codebase.
        """
        from marketcheck.validators.structural import NullValues

        ds = _make_dataset(_bar(high=float("nan")))

        assert NullValues().validate(ds, _make_context()).status == Status.PASS
        assert OhlcRangeViolation().validate(ds, _make_context()).status == Status.PASS

        result = ImpossibleValues().validate(ds, _make_context())
        assert result.status == Status.FAIL
        assert "high is NaN" in result.details["violations"][0]["failed_checks"]

    def test_fail_nan_elsewhere_is_diagnosed_correctly_not_as_geometry(self) -> None:
        """For NaN in open/low/close, OhlcRangeViolation does trip — but reports a
        GEOMETRY failure ('high < close'), misdiagnosing the problem. This rule
        names the actual cause, which is the other half of why the explicit NaN
        check matters."""
        ds = _make_dataset(_bar(close=float("nan")))

        geometry = OhlcRangeViolation().validate(ds, _make_context())
        assert geometry.status == Status.FAIL
        assert geometry.details["violations"][0]["failed_checks"] == ["high < close"]

        result = ImpossibleValues().validate(ds, _make_context())
        assert result.status == Status.FAIL
        assert "close is NaN" in result.details["violations"][0]["failed_checks"]

    def test_nan_evades_null_values_in_all_price_columns(self) -> None:
        """NullValues passes NaN in every price column, confirming NaN is not a
        null as far as that rule is concerned."""
        from marketcheck.validators.structural import NullValues

        for col in ("open", "high", "low", "close"):
            ds = _make_dataset(_bar(**{("open_" if col == "open" else col): float("nan")}))
            assert NullValues().validate(ds, _make_context()).status == Status.PASS
            assert ImpossibleValues().validate(ds, _make_context()).status == Status.FAIL

    def test_fail_nan_from_unparseable_csv_value(self) -> None:
        """A CSV containing the literal text 'NaN' reaches validation as a real
        NaN via coerce_dtypes, so this is a genuine ingestion path."""
        from marketcheck.ingestion.schema import coerce_dtypes

        raw = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 2, 9, 30)],
                "open": ["NaN"],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": [10],
            }
        )
        result = ImpossibleValues().validate(
            _make_dataset(coerce_dtypes(raw)), _make_context()
        )

        assert result.status == Status.FAIL
        assert "open is NaN" in result.details["violations"][0]["failed_checks"]

    # --- FAIL cases: negative volume ------------------------------------------

    def test_fail_negative_volume(self) -> None:
        """A negative share count is meaningless -> FAIL."""
        df = _bar().with_columns(pl.lit(-100, dtype=pl.Int64).alias("volume"))
        result = ImpossibleValues().validate(_make_dataset(df), _make_context())

        assert result.status == Status.FAIL
        assert "volume < 0" in result.details["violations"][0]["failed_checks"]

    # --- FAIL cases: no-trade bar tagging (Option B) ---------------------------

    def test_fail_no_trade_bar_is_tagged_not_four_findings(self) -> None:
        """The vendor 0/0/0/0 + volume 0 encoding is still reported (the bar is
        unusable for research) but labelled as one clear diagnosis rather than
        four redundant '<= 0' findings."""
        df = pl.DataFrame(
            {
                "open": [0.0],
                "high": [0.0],
                "low": [0.0],
                "close": [0.0],
                "volume": [0],
            },
            schema={
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
            },
        )
        result = ImpossibleValues().validate(_make_dataset(df), _make_context())

        assert result.status == Status.FAIL
        assert result.details["violations"][0]["failed_checks"] == [
            "no-trade bar (all prices zero)"
        ]
        assert result.details["no_trade_bar_count"] == 1

    def test_fail_all_zero_prices_with_volume_is_not_no_trade(self) -> None:
        """All-zero prices WITH positive volume is contradictory (trades at price
        zero?), so it is genuinely corrupt rather than a vendor convention and
        must NOT get the no-trade label."""
        df = pl.DataFrame(
            {
                "open": [0.0],
                "high": [0.0],
                "low": [0.0],
                "close": [0.0],
                "volume": [500],
            },
            schema={
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
            },
        )
        result = ImpossibleValues().validate(_make_dataset(df), _make_context())

        assert result.status == Status.FAIL
        failed = result.details["violations"][0]["failed_checks"]
        assert "no-trade bar (all prices zero)" not in failed
        assert set(failed) == {"open <= 0", "high <= 0", "low <= 0", "close <= 0"}
        assert result.details["no_trade_bar_count"] == 0

    def test_fail_no_trade_count_accurate_when_details_capped(self) -> None:
        """no_trade_bar_count is computed over the FULL violation set, so it stays
        accurate when the details list is truncated."""
        n = 20
        df = pl.DataFrame(
            {
                "open": [0.0] * n,
                "high": [0.0] * n,
                "low": [0.0] * n,
                "close": [0.0] * n,
                "volume": [0] * n,
            },
            schema={
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
            },
        )
        context = RuleContext(config=ValidationConfig(max_rows_in_details=3))
        result = ImpossibleValues().validate(_make_dataset(df), context)

        assert result.affected_rows == n
        assert len(result.details["violations"]) == 3
        assert result.details["no_trade_bar_count"] == n

    def test_fail_partial_columns_cannot_be_labelled_no_trade(self) -> None:
        """Without the full OHLC set plus volume, the no-trade pattern cannot be
        asserted, so a zero price is reported as an ordinary violation."""
        df = pl.DataFrame({"close": [0.0]}, schema={"close": pl.Float64})
        result = ImpossibleValues().validate(_make_dataset(df), _make_context())

        assert result.status == Status.FAIL
        assert result.details["violations"][0]["failed_checks"] == ["close <= 0"]
        assert result.details["no_trade_bar_count"] == 0

    # --- Mixed / counting -----------------------------------------------------

    def test_fail_distinct_rows_counted_once(self) -> None:
        """A row failing several checks counts once, mirroring OhlcRangeViolation
        and NullValues."""
        df = _bar(open_=-1.0, close=-2.0)
        result = ImpossibleValues().validate(_make_dataset(df), _make_context())

        assert result.affected_rows == 1
        assert len(result.details["violations"][0]["failed_checks"]) == 2

    def test_fail_multiple_bad_rows(self) -> None:
        """Several distinct impossible rows -> correct count and indices."""
        df = pl.DataFrame(
            {
                "open": [1.0, -1.0, 1.0, float("nan")],
                "high": [1.0, 1.0, 1.0, 1.0],
                "low": [1.0, 1.0, 1.0, 1.0],
                "close": [1.0, 1.0, 1.0, 1.0],
                "volume": [10, 10, -5, 10],
            },
            schema={
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
            },
        )
        result = ImpossibleValues().validate(_make_dataset(df), _make_context())

        assert result.affected_rows == 3
        assert {v["index"] for v in result.details["violations"]} == {1, 2, 3}

    # --- Message format checks ------------------------------------------------

    def test_fail_message_contains_count_and_check(self) -> None:
        """Message names the row count and the first failed check."""
        df = _bar(close=-3.0)
        result = ImpossibleValues().validate(_make_dataset(df), _make_context())

        assert "1 row(s)" in result.message
        assert "close <= 0" in result.message

    def test_fail_message_mentions_no_trade_bars_when_present(self) -> None:
        """When no-trade bars are found the message says so, so a reader can tell
        a vendor encoding from genuine corruption without reading details."""
        df = pl.DataFrame(
            {
                "open": [0.0],
                "high": [0.0],
                "low": [0.0],
                "close": [0.0],
                "volume": [0],
            },
            schema={
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
            },
        )
        result = ImpossibleValues().validate(_make_dataset(df), _make_context())

        assert "no-trade" in result.message.lower()

    def test_pass_message_is_informative(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """Pass message must be non-empty and describe the outcome."""
        result = ImpossibleValues().validate(_make_dataset(sample_ohlcv_df), _make_context())

        assert result.message
        assert "impossible" in result.message.lower()


# ---------------------------------------------------------------------------
# SuspiciousPriceJump — full test suite
#
# Defaults: intraday 20%, overnight 50%, split tolerance 2%.
# Division of labour: split-shaped ratios AT SESSION BOUNDARIES belong to
# CorporateActionDiscontinuity and are excluded here; the same ratio INTRADAY
# is a bad tick and stays here.
# ---------------------------------------------------------------------------

class TestSuspiciousPriceJump:
    """Tests for the SuspiciousPriceJump validation rule."""

    # --- PASS cases ---------------------------------------------------------

    def test_pass_stable_prices(self) -> None:
        """Small moves are not flagged."""
        result = SuspiciousPriceJump().validate(
            _make_dataset(_price_df([100.0, 101.0, 100.5, 102.0])), _make_context()
        )

        assert result.status == Status.PASS
        assert result.rule_id == "numerical.suspicious_price_jump"
        assert result.severity == Severity.WARNING
        assert result.affected_rows == 0
        assert result.details == {}

    def test_pass_standard_fixture(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """The standard fixture's mild moves are not suspicious."""
        result = SuspiciousPriceJump().validate(
            _make_dataset(sample_ohlcv_df), _make_context()
        )

        assert result.status == Status.PASS

    def test_pass_close_column_missing(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """No close column -> PASS (skip); MissingColumns owns absence."""
        df = sample_ohlcv_df.drop("close")
        result = SuspiciousPriceJump().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS

    def test_pass_single_row(self) -> None:
        """One row has no predecessor to compare against."""
        result = SuspiciousPriceJump().validate(
            _make_dataset(_price_df([100.0])), _make_context()
        )

        assert result.status == Status.PASS

    def test_pass_empty_dataset(self) -> None:
        """Zero rows -> PASS."""
        result = SuspiciousPriceJump().validate(
            _make_dataset(_price_df([])), _make_context()
        )

        assert result.status == Status.PASS

    def test_pass_null_close_excluded(self) -> None:
        """A null close cannot form a ratio; NullValues owns nulls."""
        result = SuspiciousPriceJump().validate(
            _make_dataset(_price_df([100.0, None, 100.0])), _make_context()
        )

        assert result.status == Status.PASS

    def test_pass_zero_previous_close_skipped(self) -> None:
        """A zero previous close would make the ratio infinite. Those rows are
        skipped — ImpossibleValues owns non-positive prices — rather than
        producing a garbage or crashing comparison."""
        result = SuspiciousPriceJump().validate(
            _make_dataset(_price_df([0.0, 100.0])), _make_context()
        )

        assert result.status == Status.PASS

    def test_pass_overnight_move_below_overnight_threshold(self) -> None:
        """A 30% overnight gap is below the 50% overnight threshold, even though
        it would exceed the 20% intraday threshold. Proves the threshold really is
        selected by transition type."""
        result = SuspiciousPriceJump().validate(
            _make_dataset(_price_df([100.0, 130.0], ["2024-01-02", "2024-01-03"])),
            _make_context(),
        )

        assert result.status == Status.PASS

    # --- Split exclusion (the division of labour) ------------------------------

    def test_pass_two_for_one_split_at_session_boundary_excluded(self) -> None:
        """A clean 2:1 split across a session boundary is CorporateActionDiscontinuity's
        finding, not ours -> excluded here so the row is never double-reported."""
        result = SuspiciousPriceJump().validate(
            _make_dataset(_price_df([100.0, 50.0], ["2024-01-02", "2024-01-03"])),
            _make_context(),
        )

        assert result.status == Status.PASS

    def test_pass_reverse_split_at_boundary_excluded(self) -> None:
        """A 1:10 reverse split (price x10) across a boundary is also excluded."""
        result = SuspiciousPriceJump().validate(
            _make_dataset(_price_df([10.0, 100.0], ["2024-01-02", "2024-01-03"])),
            _make_context(),
        )

        assert result.status == Status.PASS

    def test_pass_split_within_tolerance_excluded(self) -> None:
        """Real splits rarely land exactly on the ratio because the price also
        moves on genuine trading, so a near-2:1 ratio inside tolerance is still
        recognised as a split."""
        result = SuspiciousPriceJump().validate(
            _make_dataset(_price_df([100.0, 50.5], ["2024-01-02", "2024-01-03"])),
            _make_context(),
        )

        assert result.status == Status.PASS

    def test_warn_split_shaped_ratio_intraday_is_still_flagged(self) -> None:
        """THE key refinement: a split takes effect only between sessions, so an
        exact halving WITHIN a session is a bad tick, not a 2:1 split, and must
        remain this rule's finding.

        Without gating the exclusion on session boundaries there would be a blind
        spot at precisely the ratios corruption tends to produce (halving or
        doubling from a units/decimal error).
        """
        result = SuspiciousPriceJump().validate(
            _make_dataset(_price_df([100.0, 50.0])), _make_context()  # same date
        )

        assert result.status == Status.WARN
        assert result.details["violations"][0]["transition"] == "intraday"

    def test_warn_non_split_ratio_at_boundary_is_flagged(self) -> None:
        """A large overnight move that matches NO split ratio (here -70%) has no
        benign explanation, so it stays with this rule."""
        result = SuspiciousPriceJump().validate(
            _make_dataset(_price_df([100.0, 30.0], ["2024-01-02", "2024-01-03"])),
            _make_context(),
        )

        assert result.status == Status.WARN
        assert result.details["violations"][0]["transition"] == "overnight"

    # --- WARN cases ----------------------------------------------------------

    def test_warn_large_intraday_jump(self) -> None:
        """A 40% intraday move exceeds the 20% intraday threshold."""
        result = SuspiciousPriceJump().validate(
            _make_dataset(_price_df([100.0, 140.0])), _make_context()
        )

        assert result.status == Status.WARN
        assert result.affected_rows == 1
        v = result.details["violations"][0]
        assert v["index"] == 1
        assert v["previous_close"] == 100.0
        assert v["close"] == 140.0
        assert v["pct_change"] == 40.0
        assert v["threshold_pct"] == 20.0

    def test_warn_decimal_shift_error(self) -> None:
        """A 10x decimal-shift error (+900%) is caught even overnight, where the
        threshold is loosest — and 10.0 IS a known reverse-split ratio, so this
        also confirms that only *boundary* transitions get the split exemption."""
        result = SuspiciousPriceJump().validate(
            _make_dataset(_price_df([100.0, 1000.0])), _make_context()  # intraday
        )

        assert result.status == Status.WARN
        assert result.details["violations"][0]["pct_change"] == 900.0

    def test_warn_negative_move_reported_with_sign(self) -> None:
        """A downward move is reported as a negative percentage."""
        result = SuspiciousPriceJump().validate(
            _make_dataset(_price_df([100.0, 60.0])), _make_context()
        )

        assert result.status == Status.WARN
        assert result.details["violations"][0]["pct_change"] == -40.0

    def test_warn_multiple_jumps(self) -> None:
        """Several jumps are all reported, in index order."""
        result = SuspiciousPriceJump().validate(
            _make_dataset(_price_df([100.0, 150.0, 151.0, 90.0])), _make_context()
        )

        assert result.status == Status.WARN
        assert result.affected_rows == 2
        assert [v["index"] for v in result.details["violations"]] == [1, 3]

    def test_warn_custom_thresholds_honoured(self) -> None:
        """Thresholds are configurable, not hardcoded."""
        df = _price_df([100.0, 110.0])  # +10% intraday

        assert (
            SuspiciousPriceJump().validate(_make_dataset(df), _make_context()).status
            == Status.PASS
        )

        context = RuleContext(
            config=ValidationConfig(price_jump_intraday_threshold=0.05)
        )
        result = SuspiciousPriceJump().validate(_make_dataset(df), context)

        assert result.status == Status.WARN
        assert result.details["intraday_threshold_pct"] == 5.0

    def test_warn_no_timestamp_treated_as_overnight(self) -> None:
        """Without timestamps, session boundaries are unknowable, so the looser
        overnight threshold is applied rather than manufacturing intraday
        violations. A 30% move therefore passes."""
        df = pl.DataFrame({"close": [100.0, 130.0]}, schema={"close": pl.Float64})
        result = SuspiciousPriceJump().validate(_make_dataset(df), _make_context())

        assert result.status == Status.PASS

    def test_warn_details_capped(self) -> None:
        """details['violations'] capped; affected_rows keeps the full count."""
        closes = [100.0, 200.0] * 10  # alternating +100% / -50% intraday
        context = RuleContext(config=ValidationConfig(max_rows_in_details=3))
        result = SuspiciousPriceJump().validate(_make_dataset(_price_df(closes)), context)

        assert result.status == Status.WARN
        assert result.affected_rows > 3
        assert len(result.details["violations"]) == 3

    # --- Message format checks ------------------------------------------------

    def test_warn_message_is_review_oriented(self) -> None:
        """Like VolumeAnomaly, a large move may be a genuine market event, so the
        message invites review rather than asserting corruption."""
        result = SuspiciousPriceJump().validate(
            _make_dataset(_price_df([100.0, 140.0])), _make_context()
        )

        assert "review" in result.message.lower()
        for overclaim in ("corrupt", "invalid"):
            assert overclaim not in result.message.lower()

    def test_warn_message_contains_prices_and_change(self) -> None:
        """Message names the move and the threshold applied."""
        result = SuspiciousPriceJump().validate(
            _make_dataset(_price_df([100.0, 140.0])), _make_context()
        )

        assert "100" in result.message
        assert "140" in result.message
        assert "+40" in result.message

    def test_pass_message_is_informative(self) -> None:
        """Pass message must be non-empty and describe the outcome."""
        result = SuspiciousPriceJump().validate(
            _make_dataset(_price_df([100.0, 101.0])), _make_context()
        )

        assert result.message
        assert "jump" in result.message.lower()


# ---------------------------------------------------------------------------
# Shared split-signature helpers
# ---------------------------------------------------------------------------

class TestSplitSignatures:
    """Tests for validators/signatures.py, shared by both gap rules."""

    def test_label_recognises_common_splits(self) -> None:
        from marketcheck.validators.signatures import label_split_ratio

        assert label_split_ratio(0.5, 0.02) == "2:1 split"
        assert label_split_ratio(1 / 3, 0.02) == "3:1 split"
        assert label_split_ratio(10.0, 0.02) == "1:10 reverse split"

    def test_label_returns_none_for_unremarkable_ratio(self) -> None:
        from marketcheck.validators.signatures import label_split_ratio

        assert label_split_ratio(0.3, 0.02) is None    # -70%, no split shape
        assert label_split_ratio(0.87, 0.02) is None

    def test_label_rejects_non_positive_and_none(self) -> None:
        from marketcheck.validators.signatures import label_split_ratio

        assert label_split_ratio(None, 0.02) is None
        assert label_split_ratio(0.0, 0.02) is None
        assert label_split_ratio(-0.5, 0.02) is None

    def test_label_respects_tolerance(self) -> None:
        from marketcheck.validators.signatures import label_split_ratio

        assert label_split_ratio(0.51, 0.05) == "2:1 split"
        assert label_split_ratio(0.51, 0.001) is None
