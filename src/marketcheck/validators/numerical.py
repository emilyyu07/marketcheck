"""Numerical validation rules (4 rules)."""

import polars as pl

from marketcheck.models.dataset import CanonicalDataset
from marketcheck.models.enums import Category, Severity, Status
from marketcheck.models.result import ValidationResult
from marketcheck.validators.base import RuleContext, ValidationRule, register

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
                status=Status.PASS,
                message="All rows have valid OHLC ranges.",
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
Rule: Impossible Values  (STUB -- not yet implemented)
Checks for values that are outright impossible for equity market data, as
opposed to merely unusual:
  - any OHLC price <= 0   (a zero or negative price is not a real trade)
  - volume < 0            (a negative share count is meaningless)

Scope rationale -- why this is one rule, and why it is separate from others:
- Separate from `OhlcRangeViolation` because that rule checks *relative
  geometry* within a bar. A bar like open=-5, high=-1, low=-10, close=-3
  satisfies every geometry invariant while being economically impossible, so
  the two detect genuinely different pathologies and deserve distinct rule_ids
  and `details` shapes.
- Separate from `VolumeAnomaly` because that rule is a *statistical* detector
  whose findings are inherently ambiguous (a spike may be a real market event),
  so it carries WARNING. Negative volume is unambiguous and deserves CRITICAL.
  `ValidationRule.default_severity` is a single class-level attribute -- one
  severity per rule -- so an unambiguous CRITICAL finding cannot share a rule
  with an ambiguous WARNING one without mis-reporting one of them. That
  architectural constraint is what forced this split.
- Grouping non-positive prices together with negative volume IS appropriate:
  both are "this value is impossible" checks sharing one severity (CRITICAL)
  and one remedy (the data is wrong; go back to the source).

Deliberately NOT covered:
- `volume == 0`, which is legitimate for illiquid names, halted trading, or a
  session with no trades -- ambiguous, so not an "impossible value".

This is the 14th rule, added beyond the original 13 in the scaffold spec --
the same "separate rule rather than retrofit a second concern" reasoning
PROJECT_STATUS.md already floated for a possible `structural.null_timestamps`.
Implementation should follow `OhlcRangeViolation`'s shape closely: vectorized
`any_horizontal` over per-column predicates, distinct-row `affected_rows`,
`details` capped at `max_rows_in_details`, CRITICAL -> FAIL.
'''
@register
class ImpossibleValues(ValidationRule):
    rule_id = "numerical.impossible_values"
    rule_name = "Impossible Values"
    category = Category.NUMERICAL
    default_severity = Severity.CRITICAL

    def validate(self, dataset: CanonicalDataset, context: RuleContext) -> ValidationResult:
        raise NotImplementedError("TODO: implement numerical.impossible_values")


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
                status=Status.PASS,
                message="No volume column present; skipped.",
            )

        if df.height == 0:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=self.default_severity,
                status=Status.PASS,
                message="No volume anomalies detected.",
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


@register
class SuspiciousPriceJump(ValidationRule):
    rule_id = "numerical.suspicious_price_jump"
    rule_name = "Suspicious Price Jump"
    category = Category.NUMERICAL
    default_severity = Severity.WARNING

    def validate(self, dataset: CanonicalDataset, context: RuleContext) -> ValidationResult:
        raise NotImplementedError("TODO: implement numerical.suspicious_price_jump")
