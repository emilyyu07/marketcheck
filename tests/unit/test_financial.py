"""Tests for financial validation rules."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from marketcheck.models.config import ValidationConfig
from marketcheck.models.dataset import CanonicalDataset
from marketcheck.models.enums import Category, Severity, Status
from marketcheck.validators import REGISTRY
from marketcheck.validators.base import RuleContext
from marketcheck.validators.financial import CorporateActionDiscontinuity
from marketcheck.validators.numerical import SuspiciousPriceJump


def _make_dataset(df: pl.DataFrame) -> CanonicalDataset:
    """Wrap a DataFrame in a CanonicalDataset with a dummy source path."""
    return CanonicalDataset(df=df, source_path="test.csv", row_count=len(df))


def _make_context() -> RuleContext:
    """Return a default RuleContext."""
    return RuleContext()


def _daily(closes: list[float | None], volumes: list[int] | None = None) -> pl.DataFrame:
    """One row per consecutive calendar day, so every transition is a session boundary."""
    start = datetime(2024, 1, 2, 10, 0)
    data: dict[str, object] = {
        "timestamp": [start + timedelta(days=i) for i in range(len(closes))],
        "close": closes,
    }
    schema: dict[str, object] = {"timestamp": pl.Datetime("us"), "close": pl.Float64}
    if volumes is not None:
        data["volume"] = volumes
        schema["volume"] = pl.Int64
    return pl.DataFrame(data, schema=schema)  # type: ignore[arg-type]


def _intraday(closes: list[float]) -> pl.DataFrame:
    """All rows on one date, so NO transition is a session boundary."""
    start = datetime(2024, 1, 2, 10, 0)
    return pl.DataFrame(
        {
            "timestamp": [start + timedelta(minutes=i) for i in range(len(closes))],
            "close": closes,
        },
        schema={"timestamp": pl.Datetime("us"), "close": pl.Float64},
    )


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------

class TestFinancialRulesRegistered:
    def test_rule_is_registered(self) -> None:
        assert CorporateActionDiscontinuity in REGISTRY

    def test_rule_has_correct_category(self) -> None:
        assert CorporateActionDiscontinuity().category == Category.FINANCIAL

    def test_rule_severity_is_info(self) -> None:
        """INFO: unadjusted data is not corrupt, merely raw."""
        assert CorporateActionDiscontinuity().default_severity == Severity.INFO


# ---------------------------------------------------------------------------
# CorporateActionDiscontinuity — full test suite
# ---------------------------------------------------------------------------

class TestCorporateActionDiscontinuity:
    """Tests for the CorporateActionDiscontinuity validation rule."""

    # --- PASS cases ---------------------------------------------------------

    def test_pass_stable_prices(self) -> None:
        """Ordinary daily moves match no split signature."""
        result = CorporateActionDiscontinuity().validate(
            _make_dataset(_daily([100.0, 101.0, 99.5, 100.2])), _make_context()
        )

        assert result.status == Status.PASS
        assert result.rule_id == "financial.corporate_action_discontinuity"
        assert result.affected_rows == 0
        assert result.details == {}

    def test_pass_standard_fixture(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """The standard intraday fixture has no boundaries and no split ratios."""
        result = CorporateActionDiscontinuity().validate(
            _make_dataset(sample_ohlcv_df), _make_context()
        )

        assert result.status == Status.PASS

    def test_skip_close_column_missing(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """No close column -> skip; MissingColumns owns absence."""
        result = CorporateActionDiscontinuity().validate(
            _make_dataset(sample_ohlcv_df.drop("close")), _make_context()
        )

        assert result.status == Status.SKIP

    def test_skip_timestamp_column_missing(self) -> None:
        """Without timestamps, sessions can't be identified. The rule must skip
        rather than guess, since a split only occurs between sessions."""
        df = pl.DataFrame({"close": [100.0, 50.0]}, schema={"close": pl.Float64})
        result = CorporateActionDiscontinuity().validate(_make_dataset(df), _make_context())

        assert result.status == Status.SKIP

    def test_skip_single_row(self) -> None:
        """One row cannot form a ratio."""
        result = CorporateActionDiscontinuity().validate(
            _make_dataset(_daily([100.0])), _make_context()
        )

        assert result.status == Status.SKIP

    def test_skip_empty_dataset(self) -> None:
        """Zero rows -> PASS."""
        result = CorporateActionDiscontinuity().validate(
            _make_dataset(_daily([])), _make_context()
        )

        assert result.status == Status.SKIP

    def test_pass_split_ratio_intraday_is_not_a_split(self) -> None:
        """A split takes effect only between sessions, so an exact halving WITHIN
        a session is not a corporate action and must not be claimed as one.
        (SuspiciousPriceJump owns that row instead.)"""
        result = CorporateActionDiscontinuity().validate(
            _make_dataset(_intraday([100.0, 50.0])), _make_context()
        )

        assert result.status == Status.PASS

    def test_pass_large_move_not_matching_any_split(self) -> None:
        """A -70% overnight collapse matches no split ratio, so it is not
        attributed to a corporate action."""
        result = CorporateActionDiscontinuity().validate(
            _make_dataset(_daily([100.0, 30.0])), _make_context()
        )

        assert result.status == Status.PASS

    def test_pass_null_close_excluded(self) -> None:
        """Null closes cannot form a ratio; NullValues owns nulls."""
        result = CorporateActionDiscontinuity().validate(
            _make_dataset(_daily([100.0, None, 50.0])), _make_context()
        )

        assert result.status == Status.PASS

    def test_pass_zero_close_skipped(self) -> None:
        """A zero close would make the ratio meaningless or infinite;
        ImpossibleValues owns non-positive prices."""
        result = CorporateActionDiscontinuity().validate(
            _make_dataset(_daily([0.0, 50.0])), _make_context()
        )

        assert result.status == Status.PASS

    def test_pass_dividend_sized_drop_not_detected(self) -> None:
        """Documents the honest limitation: a ~1.5% dividend drop is
        indistinguishable from ordinary movement, so dividends are undetectable
        by ratio matching. This rule detects SPLITS only, despite its name."""
        result = CorporateActionDiscontinuity().validate(
            _make_dataset(_daily([100.0, 98.5])), _make_context()
        )

        assert result.status == Status.PASS

    # --- WARN cases ----------------------------------------------------------

    def test_warn_two_for_one_split(self) -> None:
        """The canonical case: price halves across a session boundary."""
        result = CorporateActionDiscontinuity().validate(
            _make_dataset(_daily([100.0, 50.0])), _make_context()
        )

        assert result.status == Status.WARN
        assert result.severity == Severity.INFO
        assert result.affected_rows == 1
        v = result.details["violations"][0]
        assert v["index"] == 1
        assert v["previous_close"] == 100.0
        assert v["close"] == 50.0
        assert v["ratio"] == 0.5
        assert v["pct_change"] == -50.0
        assert v["inferred_action"] == "2:1 split"

    def test_warn_three_for_two_split_below_jump_threshold(self) -> None:
        """A 3:2 split is only -33.33%, BELOW SuspiciousPriceJump's 50% overnight
        threshold. Detecting it here is exactly why this rule has no magnitude
        threshold — with one, this real split would be reported by neither rule."""
        result = CorporateActionDiscontinuity().validate(
            _make_dataset(_daily([150.0, 100.0])), _make_context()
        )

        assert result.status == Status.WARN
        assert result.details["violations"][0]["inferred_action"] == "3:2 split"

    def test_warn_four_for_three_split_smallest_signature(self) -> None:
        """4:3 (-25%) is the smallest-magnitude signature and must still be caught."""
        result = CorporateActionDiscontinuity().validate(
            _make_dataset(_daily([100.0, 75.0])), _make_context()
        )

        assert result.status == Status.WARN
        assert result.details["violations"][0]["inferred_action"] == "4:3 split"

    def test_warn_reverse_split(self) -> None:
        """A 1:10 reverse split multiplies the price tenfold."""
        result = CorporateActionDiscontinuity().validate(
            _make_dataset(_daily([10.0, 100.0])), _make_context()
        )

        assert result.status == Status.WARN
        assert result.details["violations"][0]["inferred_action"] == "1:10 reverse split"

    def test_warn_split_within_tolerance(self) -> None:
        """Real splits rarely land exactly on the ratio because the price also
        moves on genuine trading, so a near-2:1 ratio inside tolerance matches."""
        result = CorporateActionDiscontinuity().validate(
            _make_dataset(_daily([100.0, 50.5])), _make_context()
        )

        assert result.status == Status.WARN
        assert result.details["violations"][0]["inferred_action"] == "2:1 split"

    def test_warn_tolerance_is_configurable(self) -> None:
        """A ratio outside a tightened tolerance stops matching."""
        df = _daily([100.0, 50.5])  # ratio 0.505, 1% off 0.5

        assert (
            CorporateActionDiscontinuity().validate(_make_dataset(df), _make_context()).status
            == Status.WARN
        )

        context = RuleContext(config=ValidationConfig(split_ratio_tolerance=0.001))
        result = CorporateActionDiscontinuity().validate(_make_dataset(df), context)

        assert result.status == Status.PASS

    def test_warn_volume_ratio_reported_as_evidence(self) -> None:
        """Volume corroborates but never filters: a 2:1 split roughly doubles
        share volume, and that ratio is surfaced for the reader to weigh."""
        result = CorporateActionDiscontinuity().validate(
            _make_dataset(_daily([100.0, 50.0], volumes=[1000, 2000])), _make_context()
        )

        assert result.status == Status.WARN
        assert result.details["violations"][0]["volume_ratio"] == 2.0

    def test_warn_split_detected_even_when_volume_uncooperative(self) -> None:
        """Volume is NOT required: an unchanged (or restated) volume must not
        suppress a genuine split, which is why corroboration is evidence only."""
        result = CorporateActionDiscontinuity().validate(
            _make_dataset(_daily([100.0, 50.0], volumes=[1000, 1000])), _make_context()
        )

        assert result.status == Status.WARN
        assert result.details["violations"][0]["volume_ratio"] == 1.0

    def test_warn_volume_ratio_omitted_when_no_volume_column(self) -> None:
        """Without a volume column the field is simply absent, not null."""
        result = CorporateActionDiscontinuity().validate(
            _make_dataset(_daily([100.0, 50.0])), _make_context()
        )

        assert "volume_ratio" not in result.details["violations"][0]

    def test_warn_multiple_splits_tallied(self) -> None:
        """Several splits are reported and tallied by inferred action."""
        # 100 -> 50 (2:1), then 50 -> 25 (2:1) on later days.
        result = CorporateActionDiscontinuity().validate(
            _make_dataset(_daily([100.0, 50.0, 25.0])), _make_context()
        )

        assert result.status == Status.WARN
        assert result.affected_rows == 2
        assert result.details["inferred_action_counts"] == {"2:1 split": 2}

    def test_warn_counts_accurate_when_details_capped(self) -> None:
        """inferred_action_counts is computed over the FULL candidate set, so it
        stays accurate when the details list is truncated."""
        closes = [100.0 / (2**i) for i in range(8)]  # repeated 2:1 halvings
        context = RuleContext(config=ValidationConfig(max_rows_in_details=2))
        result = CorporateActionDiscontinuity().validate(_make_dataset(_daily(closes)), context)

        assert result.affected_rows == 7
        assert len(result.details["violations"]) == 2
        assert result.details["inferred_action_counts"] == {"2:1 split": 7}

    def test_warn_details_echo_tolerance(self) -> None:
        """The tolerance used is echoed so a reader can judge match strength."""
        result = CorporateActionDiscontinuity().validate(
            _make_dataset(_daily([100.0, 50.0])), _make_context()
        )

        assert result.details["split_ratio_tolerance"] == 0.02

    # --- Message wording ------------------------------------------------------

    def test_warn_message_says_unadjusted_not_corrupt(self) -> None:
        """A finding means the data is raw, not broken — and the inference is
        unverifiable without a corporate-actions feed, so the wording must hedge
        ('consistent with') rather than assert that a split occurred."""
        result = CorporateActionDiscontinuity().validate(
            _make_dataset(_daily([100.0, 50.0])), _make_context()
        )

        lowered = result.message.lower()
        assert "unadjusted" in lowered
        assert "consistent with" in lowered
        for overclaim in ("corrupt", "invalid", "error"):
            assert overclaim not in lowered

    def test_warn_message_names_inferred_action_and_prices(self) -> None:
        result = CorporateActionDiscontinuity().validate(
            _make_dataset(_daily([100.0, 50.0])), _make_context()
        )

        assert "2:1 split" in result.message
        assert "100" in result.message
        assert "50" in result.message

    def test_pass_message_is_informative(self) -> None:
        result = CorporateActionDiscontinuity().validate(
            _make_dataset(_daily([100.0, 101.0])), _make_context()
        )

        assert result.message
        assert "corporate-action" in result.message.lower()


# ---------------------------------------------------------------------------
# Mutual exclusion with SuspiciousPriceJump
#
# The pair must partition large moves: every row is reported by at most one of
# them, and a large boundary move is never dropped by both.
# ---------------------------------------------------------------------------

class TestGapRulePartition:
    """The two gap-detecting rules must never double-report or both stay silent."""

    def _statuses(self, df: pl.DataFrame) -> tuple[Status, Status]:
        ds = _make_dataset(df)
        return (
            SuspiciousPriceJump().validate(ds, _make_context()).status,
            CorporateActionDiscontinuity().validate(ds, _make_context()).status,
        )

    def test_split_at_boundary_owned_only_by_financial_rule(self) -> None:
        """2:1 split overnight: corporate-action rule only."""
        jump, action = self._statuses(_daily([100.0, 50.0]))

        assert jump == Status.PASS
        assert action == Status.WARN

    def test_non_split_large_move_owned_only_by_jump_rule(self) -> None:
        """-70% overnight matches no split: jump rule only."""
        jump, action = self._statuses(_daily([100.0, 30.0]))

        assert jump == Status.WARN
        assert action == Status.PASS

    def test_split_shaped_intraday_move_owned_only_by_jump_rule(self) -> None:
        """Identical -50% ratio, but intraday, so it is a bad tick: jump rule only."""
        jump, action = self._statuses(_intraday([100.0, 50.0]))

        assert jump == Status.WARN
        assert action == Status.PASS

    def test_small_split_is_not_missed_by_both(self) -> None:
        """A 3:2 split (-33%) sits below the jump rule's 50% overnight threshold.
        Regression guard for the gap that a magnitude threshold here would create:
        exactly one rule must still report it."""
        jump, action = self._statuses(_daily([150.0, 100.0]))

        assert jump == Status.PASS
        assert action == Status.WARN

    def test_ordinary_move_reported_by_neither(self) -> None:
        """A benign move belongs to neither rule."""
        jump, action = self._statuses(_daily([100.0, 101.0]))

        assert jump == Status.PASS
        assert action == Status.PASS
