"""Temporal validation rules (4 rules)."""

import polars as pl

from marketcheck.models.dataset import CanonicalDataset
from marketcheck.models.enums import Category, Severity, Status
from marketcheck.models.result import ValidationResult
from marketcheck.validators.base import RuleContext, ValidationRule, register

'''
Rule: Missing Trading Sessions
Checks whether any NYSE trading session inside the dataset's own date range has
no rows at all.

Mechanics: expected sessions from context.calendar.valid_sessions(start, end)
MINUS the set of dates actually present in `timestamp` -> anything left over is
a session the exchange was open for but the data doesn't cover.
'''
@register
class MissingSessions(ValidationRule):
    rule_id = "temporal.missing_sessions"
    rule_name = "Missing Trading Sessions"
    category = Category.TEMPORAL
    default_severity = Severity.WARNING

    def validate(self, dataset: CanonicalDataset, context: RuleContext) -> ValidationResult:
        df = dataset.df

        # MissingColumns owns absent columns; skip rather than crash.
        if "timestamp" not in df.columns:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=self.default_severity,
                status=Status.PASS,
                message="No timestamp column present; skipped.",
            )

        # NullValues owns nulls; drop them before deriving the date range so a
        # null can't collapse the range or masquerade as a covered session.
        dates_present = set(
            df.select(pl.col("timestamp").dt.date().alias("d"))
            .drop_nulls()
            .to_series()
            .unique()
            .to_list()
        )

        # No usable timestamps -> no range to diff over.
        if not dates_present:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=self.default_severity,
                status=Status.PASS,
                message="No trading sessions missing.",
            )

        start = min(dates_present)
        end = max(dates_present)

        # Single schedule build for the whole range (already chronological).
        expected_sessions = context.calendar.valid_sessions(start, end)
        missing = [d for d in expected_sessions if d not in dates_present]

        if not missing:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=self.default_severity,
                status=Status.PASS,
                message="No trading sessions missing.",
            )

        capped = missing[: context.config.max_rows_in_details]
        message = (
            f"{len(missing)} trading session(s) missing between {start} and {end} "
            f"(first: {missing[0]})."
        )

        return ValidationResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            category=self.category,
            severity=self.default_severity,
            status=Status.WARN,
            message=message,
            details={
                "missing_sessions": [str(d) for d in capped],
                "missing_count": len(missing),
                "expected_session_count": len(expected_sessions),
                "present_session_count": len(dates_present),
                "range_start": str(start),
                "range_end": str(end),
            },
            affected_rows=0,
        )


@register
class GapsWithinSession(ValidationRule):
    rule_id = "temporal.gaps_within_session"
    rule_name = "Gaps Within Session"
    category = Category.TEMPORAL
    default_severity = Severity.WARNING

    def validate(self, dataset: CanonicalDataset, context: RuleContext) -> ValidationResult:
        raise NotImplementedError("TODO: implement temporal.gaps_within_session")


'''
Rule: Outside Trading Hours
Checks whether any row's timestamp falls outside NYSE regular trading hours
([09:30:00, 16:00:00) ET, half-open interval), by wall-clock time-of-day only.
'''
@register
class OutsideTradingHours(ValidationRule):
    rule_id = "temporal.outside_trading_hours"
    rule_name = "Data Outside Trading Hours"
    category = Category.TEMPORAL
    default_severity = Severity.INFO

    def validate(self, dataset: CanonicalDataset, context: RuleContext) -> ValidationResult:
        df = dataset.df

        # MissingColumns owns absent columns; skip rather than crash or double-report.
        if "timestamp" not in df.columns:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=self.default_severity,
                status=Status.PASS,
                message="No timestamp column present; skipped.",
            )

        # Nothing to check with zero rows. Unlike the pairwise structural
        # rules, this rule inspects each row independently, so even a single
        # row is meaningfully checkable -- no "< 2 rows" guard needed here.
        if df.height == 0:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=self.default_severity,
                status=Status.PASS,
                message="No data outside regular trading hours.",
            )

        regular_open = context.calendar.regular_open()
        regular_close = context.calendar.regular_close()

        # Vectorized pass: extract wall-clock time-of-day and keep only rows
        # outside the half-open [regular_open, regular_close) interval. Null
        # timestamps produce a null time_of_day, and comparisons against
        # null evaluate to null (not True) in the filter, so they're
        # excluded automatically -- NullValues owns reporting them.
        violations_df = (
            df.select(
                pl.int_range(0, df.height).alias("index"),
                pl.col("timestamp"),
                pl.col("timestamp").dt.time().alias("time_of_day"),
            )
            .filter(
                (pl.col("time_of_day") < regular_open)
                | (pl.col("time_of_day") >= regular_close)
            )
        )

        affected_rows = violations_df.height

        if affected_rows == 0:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=self.default_severity,
                status=Status.PASS,
                message="No data outside regular trading hours.",
            )

        capped = violations_df.head(context.config.max_rows_in_details)
        violations = [
            {
                "index": row["index"],
                "timestamp": row["timestamp"],
                "time_of_day": row["time_of_day"],
            }
            for row in capped.iter_rows(named=True)
        ]

        first = violations[0]
        message = (
            f"{affected_rows} row(s) outside regular trading hours "
            f"({regular_open}-{regular_close} ET) "
            f"(first at index {first['index']}: {first['time_of_day']})."
        )

        return ValidationResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            category=self.category,
            severity=self.default_severity,
            status=Status.WARN,
            message=message,
            details={"violations": violations},
            affected_rows=affected_rows,
        )


@register
class TimezoneInconsistency(ValidationRule):
    rule_id = "temporal.timezone_inconsistency"
    rule_name = "Timezone Inconsistency"
    category = Category.TEMPORAL
    default_severity = Severity.CRITICAL

    def validate(self, dataset: CanonicalDataset, context: RuleContext) -> ValidationResult:
        raise NotImplementedError("TODO: implement temporal.timezone_inconsistency")
