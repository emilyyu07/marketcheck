"""Numerical validation rules (4 rules)."""

import polars as pl

from marketcheck.models.dataset import CanonicalDataset
from marketcheck.models.enums import Category, Severity, Status
from marketcheck.models.result import ValidationResult
from marketcheck.validators.base import RuleContext, ValidationRule, register
from marketcheck.validators.signatures import matches_split_ratio_expr

# The four price columns whose internal geometry OhlcRangeViolation validates.
# `volume` is deliberately excluded -- VolumeAnomaly owns volume.
OHLC_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")

'''
The five elementary ways a bar's OHLC geometry can be malformed, as
(human-readable name, left column, comparison, right column). These are the
expansion of three invariants: high >= low, high >= max(open, close), and
low <= min(open, close). Kept expanded rather than collapsed into max()/min()
so a report can name exactly which relationship broke -- "high < low" is a
different pathology (often a swapped-column bug) than "close > high" (often a
bad tick or an adjustment artifact).
'''

_OHLC_CHECKS: tuple[tuple[str, str, str], ...] = (
    ("high < low", "high", "low"),
    ("high < open", "high", "open"),
    ("high < close", "high", "close"),
    ("low > open", "low", "open"),
    ("low > close", "low", "close"),
)


'''
Rule: OHLC Range Violation
Checks that each row's OHLC values form a well-formed bar:
  - high >= low                     (range is not inverted)
  - high >= open and high >= close  (high really is the maximum)
  - low  <= open and low  <= close  (low really is the minimum)
'''
@register
class OhlcRangeViolation(ValidationRule):
    rule_id = "numerical.ohlc_range_violation"
    rule_name = "OHLC Range Violation"
    category = Category.NUMERICAL
    default_severity = Severity.CRITICAL

    def validate(self, dataset: CanonicalDataset, context: RuleContext) -> ValidationResult:
        df = dataset.df

        # Build only the checks whose both operand columns exist.
        # MissingColumns owns reporting absent columns.
        applicable = [
            (name, left, right)
            for name, left, right in _OHLC_CHECKS
            if left in df.columns and right in df.columns
        ]

        # Nothing comparable (fewer than two OHLC columns present, or zero rows).
        if not applicable or df.height == 0:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=self.default_severity,
                status=Status.SKIP,
                message="No OHLC columns to compare, or no rows; skipped.",
            )

        # One boolean column per check. `high < low` style names encode the
        # violating condition, so a True value means "this check failed".
        check_exprs = [
            (
                pl.col(left) < pl.col(right)
                if "<" in name
                else pl.col(left) > pl.col(right)
            ).alias(name)
            for name, left, right in applicable
        ]

        present_ohlc = [c for c in OHLC_COLUMNS if c in df.columns]

        flagged = df.select(
            pl.int_range(0, df.height).alias("index"),
            *[pl.col(c) for c in present_ohlc],
            *check_exprs,
        )

        check_names = [name for name, _, _ in applicable]

        # A row is a violation if ANY check failed. Counted once per row, not
        # once per failed check -- mirrors NullValues' distinct-row semantics.
        violations_df = flagged.filter(pl.any_horizontal([pl.col(n) for n in check_names]))

        affected_rows = violations_df.height

        if affected_rows == 0:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=self.default_severity,
                status=Status.PASS,
                message="All rows have valid OHLC ranges.",
            )

        # Python-side work happens only on the capped subset, never the full set.
        capped = violations_df.head(context.config.max_rows_in_details)
        violations = []
        for row in capped.iter_rows(named=True):
            violations.append(
                {
                    "index": row["index"],
                    **{c: row[c] for c in present_ohlc},
                    "failed_checks": [n for n in check_names if row[n]],
                }
            )

        first = violations[0]
        message = (
            f"{affected_rows} row(s) have invalid OHLC ranges "
            f"(first at index {first['index']}: {', '.join(first['failed_checks'])})."
        )

        return ValidationResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            category=self.category,
            severity=self.default_severity,
            status=Status.FAIL,
            message=message,
            details={"violations": violations},
            affected_rows=affected_rows,
        )


'''
Rule: Impossible Values
Checks for values that are outright impossible for equity market data, as
opposed to merely unusual:
  - any OHLC price <= 0    (a zero or negative price is not a real trade)
  - any OHLC price is NaN  (definitionally not a number, let alone a price)
  - volume < 0             (a negative share count is meaningless)
'''
@register
class ImpossibleValues(ValidationRule):
    rule_id = "numerical.impossible_values"
    rule_name = "Impossible Values"
    category = Category.NUMERICAL
    default_severity = Severity.CRITICAL

    # Label used when a row matches the vendor no-trade encoding.
    NO_TRADE_LABEL = "no-trade bar (all prices zero)"

    def validate(self, dataset: CanonicalDataset, context: RuleContext) -> ValidationResult:
        df = dataset.df

        present_prices = [c for c in OHLC_COLUMNS if c in df.columns]
        has_volume = "volume" in df.columns

        # (check name, expression) pairs, built only for columns present.
        checks: list[tuple[str, pl.Expr]] = []
        for col in present_prices:
            checks.append((f"{col} <= 0", pl.col(col) <= 0))
            # NaN applies to float columns only; volume is Int64 and cannot be NaN.
            if df.schema[col].is_float():
                checks.append((f"{col} is NaN", pl.col(col).is_nan()))
        if has_volume:
            checks.append(("volume < 0", pl.col("volume") < 0))

        if not checks or df.height == 0:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=self.default_severity,
                status=Status.SKIP,
                message="No price or volume columns to inspect, or no rows; skipped.",
            )

        check_names = [name for name, _ in checks]
        reported_cols = present_prices + (["volume"] if has_volume else [])

        # The no-trade pattern requires the full OHLC set AND volume, so a
        # partial-column dataset can't be mislabelled as a no-trade bar. Rows
        # with all-zero prices but POSITIVE volume are contradictory rather than
        # a vendor convention, so volume == 0 is part of the pattern.
        if len(present_prices) == len(OHLC_COLUMNS) and has_volume:
            no_trade_expr = pl.all_horizontal(
                [pl.col(c) == 0 for c in present_prices] + [pl.col("volume") == 0]
            )
        else:
            no_trade_expr = pl.lit(False)  # noqa: FBT003

        flagged = df.select(
            pl.int_range(0, df.height).alias("index"),
            *[pl.col(c) for c in reported_cols],
            *[expr.alias(name) for name, expr in checks],
            no_trade_expr.alias("_no_trade"),
        )

        violations_df = flagged.filter(pl.any_horizontal([pl.col(n) for n in check_names]))
        affected_rows = violations_df.height

        if affected_rows == 0:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=self.default_severity,
                status=Status.PASS,
                message="No impossible values found.",
            )

        capped = violations_df.head(context.config.max_rows_in_details)
        violations = []
        no_trade_count = 0
        for row in capped.iter_rows(named=True):
            if row["_no_trade"]:
                failed = [self.NO_TRADE_LABEL]
            else:
                failed = [n for n in check_names if row[n]]
            violations.append(
                {
                    "index": row["index"],
                    **{c: row[c] for c in reported_cols},
                    "failed_checks": failed,
                }
            )

        # Counted over the full violation set, not just the capped sample, so the
        # summary stays accurate when details are truncated.
        if "_no_trade" in violations_df.columns:
            no_trade_count = violations_df.select(pl.col("_no_trade").sum()).item()

        first = violations[0]
        message = (
            f"{affected_rows} row(s) contain impossible values "
            f"(first at index {first['index']}: {', '.join(first['failed_checks'])})."
        )
        if no_trade_count:
            message += (
                f" {no_trade_count} of these are no-trade bars "
                f"(all prices zero with zero volume)."
            )

        return ValidationResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            category=self.category,
            severity=self.default_severity,
            status=Status.FAIL,
            message=message,
            details={
                "violations": violations,
                "no_trade_bar_count": no_trade_count,
            },
            affected_rows=affected_rows,
        )


'''
Rule: Volume Anomaly
Flags bars whose volume is implausibly large relative to nearby bars, as a
CANDIDATE FOR HUMAN REVIEW (not a definitive corrupt data verdict))

**Core limitation, stated up front**: this rule cannot distinguish a data error
from a genuine market event (thus false positives are acceptable and expected). 

Method: flag when `volume > k * rolling_median(volume, window)`, centered.
'''
@register
class VolumeAnomaly(ValidationRule):
    rule_id = "numerical.volume_anomaly"
    rule_name = "Volume Anomaly"
    category = Category.NUMERICAL
    default_severity = Severity.WARNING

    def validate(self, dataset: CanonicalDataset, context: RuleContext) -> ValidationResult:
        df = dataset.df

        # MissingColumns owns absent columns; skip rather than crash.
        if "volume" not in df.columns:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=self.default_severity,
                status=Status.SKIP,
                message="No volume column present; skipped.",
            )

        if df.height == 0:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=self.default_severity,
                status=Status.SKIP,
                message="No rows to inspect; skipped.",
            )

        multiplier = context.config.volume_anomaly_multiplier
        window = context.config.volume_anomaly_window

        # Centered rolling median baseline. min_samples=1 keeps edge bars
        # auditable instead of null (no blind spots), at the cost of noisier
        # baselines there.
        baseline = (
            pl.col("volume")
            .cast(pl.Float64)
            .rolling_median(window_size=window, min_samples=1, center=True)
        )

        flagged = df.select(
            pl.int_range(0, df.height).alias("index"),
            pl.col("volume"),
            baseline.alias("rolling_median"),
        ).with_columns(
            (pl.col("volume") / pl.col("rolling_median")).alias("ratio"),
        )

        # The zero-median guard is what makes this safe on illiquid/halted data.
        violations_df = flagged.filter(
            (pl.col("rolling_median") > 0)
            & (pl.col("volume") > multiplier * pl.col("rolling_median"))
        )

        affected_rows = violations_df.height

        if affected_rows == 0:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=self.default_severity,
                status=Status.PASS,
                message="No volume anomalies detected.",
            )

        capped = violations_df.head(context.config.max_rows_in_details)
        violations = [
            {
                "index": row["index"],
                "volume": row["volume"],
                "rolling_median": row["rolling_median"],
                "ratio": round(row["ratio"], 2),
            }
            for row in capped.iter_rows(named=True)
        ]

        first = violations[0]
        # Worded as "review", not "corrupt": these are candidates, and a genuine
        # market event is indistinguishable from an error at this layer.
        # Violations stay in index order, consistent with every other rule.
        message = (
            f"{affected_rows} bar(s) have volume exceeding {multiplier:g}x the "
            f"{window}-bar rolling median and may warrant review "
            f"(first at index {first['index']}: "
            f"volume {first['volume']} vs median {first['rolling_median']:g}, "
            f"{first['ratio']:g}x)."
        )

        return ValidationResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            category=self.category,
            severity=self.default_severity,
            status=Status.WARN,
            message=message,
            details={
                "violations": violations,
                "multiplier": multiplier,
                "window": window,
            },
            affected_rows=affected_rows,
        )


'''
Rule: Suspicious Price Jump
Flags large close-to-close price moves that have no reasonable explanation, as a
candidate for human review (not a definitive corrupt data verdict).

Key points:
- cannot distynguish a data error from a genuine market event (thus 
false positives are acceptable and expected)
- only close-to-close moves are examined in v1 (known gap: a single-bar 
"wick" spike is geometrically valid and does not move close-to-close returns, 
so nothing currently detects it)
- dividends and splits are excluded, but only at session boundaries (a 
split takes effect at the start of a session, never mid-session)
'''
@register
class SuspiciousPriceJump(ValidationRule):
    rule_id = "numerical.suspicious_price_jump"
    rule_name = "Suspicious Price Jump"
    category = Category.NUMERICAL
    default_severity = Severity.WARNING

    def validate(self, dataset: CanonicalDataset, context: RuleContext) -> ValidationResult:
        df = dataset.df

        # MissingColumns owns absent columns; skip rather than crash.
        if "close" not in df.columns:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=self.default_severity,
                status=Status.SKIP,
                message="No close column present; skipped.",
            )

        # Need at least one consecutive pair to compute a move.
        if df.height < 2:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=self.default_severity,
                status=Status.SKIP,
                message="Fewer than 2 rows, so no price move can be computed; skipped.",
            )

        intraday_threshold = context.config.price_jump_intraday_threshold
        overnight_threshold = context.config.price_jump_overnight_threshold
        tolerance = context.config.split_ratio_tolerance

        # Without timestamps we cannot identify session boundaries, so treat
        # every transition as overnight (looser) rather than inventing intraday
        # violations.
        if "timestamp" in df.columns:
            date_col = pl.col("timestamp").dt.date()
            is_boundary = date_col != date_col.shift(1)
        else:
            is_boundary = pl.lit(True)  # noqa: FBT003

        prev_close = pl.col("close").shift(1)
        ratio = pl.col("close") / prev_close

        base = df.select(
            pl.int_range(0, df.height).alias("index"),
            *([pl.col("timestamp")] if "timestamp" in df.columns else []),
            prev_close.alias("previous_close"),
            pl.col("close"),
            ratio.alias("ratio"),
            is_boundary.alias("is_session_boundary"),
        ).with_columns(
            ((pl.col("ratio") - 1) * 100).alias("pct_change"),
            pl.when(pl.col("is_session_boundary"))
            .then(pl.lit(overnight_threshold))
            .otherwise(pl.lit(intraday_threshold))
            .alias("threshold"),
        )

        # A split only occurs between sessions, so the exclusion is gated on the
        # boundary flag -- intraday split-shaped ratios remain suspicious.
        looks_like_split = pl.col("is_session_boundary") & matches_split_ratio_expr(
            pl.col("ratio"), tolerance
        )

        violations_df = base.filter(
            # Guard against non-positive/zero prices: the ratio would be
            # meaningless or infinite. ImpossibleValues owns those rows.
            (pl.col("previous_close") > 0)
            & (pl.col("close") > 0)
            & ((pl.col("ratio") - 1).abs() > pl.col("threshold"))
            & ~looks_like_split
        )

        affected_rows = violations_df.height

        if affected_rows == 0:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=self.default_severity,
                status=Status.PASS,
                message="No suspicious price jumps detected.",
            )

        capped = violations_df.head(context.config.max_rows_in_details)
        violations = [
            {
                "index": row["index"],
                "previous_close": row["previous_close"],
                "close": row["close"],
                "pct_change": round(row["pct_change"], 2),
                "transition": (
                    "overnight" if row["is_session_boundary"] else "intraday"
                ),
                "threshold_pct": round(row["threshold"] * 100, 2),
            }
            for row in capped.iter_rows(named=True)
        ]

        first = violations[0]
        # Review-oriented wording: a large move may be a genuine market event.
        message = (
            f"{affected_rows} price move(s) exceed the configured jump threshold "
            f"and may warrant review (first at index {first['index']}: "
            f"{first['previous_close']:g} -> {first['close']:g}, "
            f"{first['pct_change']:+g}% {first['transition']}, "
            f"threshold {first['threshold_pct']:g}%)."
        )

        return ValidationResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            category=self.category,
            severity=self.default_severity,
            status=Status.WARN,
            message=message,
            details={
                "violations": violations,
                "intraday_threshold_pct": round(intraday_threshold * 100, 2),
                "overnight_threshold_pct": round(overnight_threshold * 100, 2),
                "split_ratio_tolerance": tolerance,
            },
            affected_rows=affected_rows,
        )
