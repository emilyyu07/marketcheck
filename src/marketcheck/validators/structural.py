"""
Structural validation rules (5 rules).

These rules check the structure of the dataset, 
including required columns, data types, and timestamp ordering.

Rule 1: MissingColumns
Rule 2: InvalidDTypes
Rule 3: UnsortedTimestamps
Rule 4:
Rule 5:
"""

import polars as pl

from marketcheck.ingestion.schema import EXPECTED_DTYPES, REQUIRED_COLUMNS
from marketcheck.models.dataset import CanonicalDataset
from marketcheck.models.enums import Category, Severity, Status
from marketcheck.models.result import ValidationResult
from marketcheck.validators.base import RuleContext, ValidationRule, register


@register
class DuplicateTimestamps(ValidationRule):
    rule_id = "structural.duplicate_timestamps"
    rule_name = "Duplicate Timestamps"
    category = Category.STRUCTURAL
    default_severity = Severity.CRITICAL

    def validate(self, dataset: CanonicalDataset, context: RuleContext) -> ValidationResult:
        raise NotImplementedError("TODO: implement structural.duplicate_timestamps")

'''
Rule 1: Missing Required Columns
Checks if the dataset contains all required columns. 
If any required columns are missing, the rule fails.
'''
@register
class MissingColumns(ValidationRule):
    rule_id = "structural.missing_columns"
    rule_name = "Missing Required Columns"
    category = Category.STRUCTURAL
    default_severity = Severity.CRITICAL

    def validate(self, dataset: CanonicalDataset, context: RuleContext) -> ValidationResult:
        missing = [col for col in REQUIRED_COLUMNS if col not in dataset.df.columns]

        if not missing:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=self.default_severity,
                status=Status.PASS,
                message="All required columns are present.",
            )

        return ValidationResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            category=self.category,
            severity=self.default_severity,
            status=Status.FAIL,
            message=f"{len(missing)} required column(s) missing: {missing}",
            details={"missing_columns": missing},
            affected_rows=0,
        )


'''
Rule 2: Invalid Data Types
Checks if the dataset columns have the expected data types.
If any column has an unexpected data type, the rule fails.
'''
@register
class InvalidDtypes(ValidationRule):
    rule_id = "structural.invalid_dtypes"
    rule_name = "Invalid Data Types"
    category = Category.STRUCTURAL
    default_severity = Severity.CRITICAL

    def validate(self, dataset: CanonicalDataset, context: RuleContext) -> ValidationResult:
        df = dataset.df

        mismatches = {
            col: {"expected": str(expected_dtype), "actual": str(df.schema[col])}
            for col, expected_dtype in EXPECTED_DTYPES.items()
            if col in df.columns and df.schema[col] != expected_dtype
        }

        if not mismatches:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=self.default_severity,
                status=Status.PASS,
                message="All column dtypes match the expected schema.",
            )

        return ValidationResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            category=self.category,
            severity=self.default_severity,
            status=Status.FAIL,
            message=f"{len(mismatches)} column(s) have unexpected dtypes: {list(mismatches)}",
            details={"mismatched_columns": mismatches},
            affected_rows=dataset.row_count,
        )


'''
Rule 3: Unsorted Timestamps
Checks if the timestamps in the dataset are sorted in strictly ascending order.
If any timestamp is not in strictly ascending order, the rule fails.
Only STRICT decreases (timestamp[i] < timestamp[i-1]) are flagged.

Key notes:
Equal consecutive timestamps (ties), absent timestamp columns, and null
timestamps are NOT flagged here.
'''
@register
class UnsortedTimestamps(ValidationRule):
    rule_id = "structural.unsorted_timestamps"
    rule_name = "Unsorted Timestamps"
    category = Category.STRUCTURAL
    default_severity = Severity.WARNING

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

        # Trivially sorted with 0 or 1 rows -- nothing to compare.
        if df.height < 2:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=self.default_severity,
                status=Status.PASS,
                message="Timestamps are sorted in ascending order.",
            )

        # Vectorized pass: pair each row with its predecessor via shift(1),
        # then keep only rows where a strict decrease occurred. Nulls on either
        # side make the comparison null (not True), so they're excluded
        # automatically without extra filtering logic.
        violations_df = (
            df.select(
                pl.int_range(0, df.height).alias("index"),
                pl.col("timestamp").alias("current_timestamp"),
                pl.col("timestamp").shift(1).alias("previous_timestamp"),
            )
            .filter(pl.col("current_timestamp") < pl.col("previous_timestamp"))
        )

        violation_count = violations_df.height

        if violation_count == 0:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=self.default_severity,
                status=Status.PASS,
                message="Timestamps are sorted in ascending order.",
            )

        capped = violations_df.head(context.config.max_rows_in_details)
        violations = [
            {
                "index": row["index"],
                "previous_timestamp": row["previous_timestamp"],
                "current_timestamp": row["current_timestamp"],
            }
            for row in capped.iter_rows(named=True)
        ]

        first = violations[0]
        message = (
            f"{violation_count} row(s) out of order "
            f"(first at index {first['index']}: "
            f"{first['previous_timestamp']} -> {first['current_timestamp']})."
        )

        return ValidationResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            category=self.category,
            severity=self.default_severity,
            status=Status.WARN,
            message=message,
            details={"violations": violations},
            affected_rows=violation_count,
        )

'''
Rule 4: Null / Missing Values
Checks if any required columns contain null or missing values.
If any required columns contain null values, the rule fails.
'''
'''
Rule 5: Null / Missing Values
Checks the 6 required OHLCV columns (timestamp, open, high, low, close, volume)
for null values.

Scope: only REQUIRED_COLUMNS are checked; any extra columns present in the
DataFrame are ignored (out of the canonical OHLCV contract this tool validates).

If a required column is entirely absent from the DataFrame, it is silently
skipped here rather than reported or causing a crash -- MissingColumns already
owns reporting column absence, and this rule must not double-report or treat
absence as "all null."

Severity is a single, uniform WARNING for the whole rule regardless of which
column(s) have nulls (this codebase's ValidationRule has one default_severity
per rule, not per-finding). A null timestamp arguably deserves harsher
treatment than a null volume, but introducing per-column severity would be a
first-of-its-kind pattern here -- deferred; see PROJECT_STATUS.md.
'''
@register
class NullValues(ValidationRule):
    rule_id = "structural.null_values"
    rule_name = "Null / Missing Values"
    category = Category.STRUCTURAL
    default_severity = Severity.WARNING

    def validate(self, dataset: CanonicalDataset, context: RuleContext) -> ValidationResult:
        df = dataset.df

        # MissingColumns owns absent columns; only check columns that exist.
        checked_cols = [col for col in REQUIRED_COLUMNS if col in df.columns]

        # Guard: if none of the required columns exist, there's nothing to
        # check -- PASS trivially rather than depending on unverified
        # pl.any_horizontal([]) behavior on an empty expression list.
        if not checked_cols:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=self.default_severity,
                status=Status.PASS,
                message="No null values found in required columns.",
            )

        # Per-column null counts -- only columns with at least one null.
        null_counts = {
            col: count
            for col in checked_cols
            if (count := df[col].null_count()) > 0
        }

        if not null_counts:
            return ValidationResult(
                rule_id=self.rule_id,
                rule_name=self.rule_name,
                category=self.category,
                severity=self.default_severity,
                status=Status.PASS,
                message="No null values found in required columns.",
            )

        # Row-level count: rows with a null in ANY checked column, counted once
        # each (not once per affected column). Single vectorized pass.
        any_null_expr = pl.any_horizontal(
            [pl.col(col).is_null() for col in checked_cols]
        )
        affected_rows = df.select(any_null_expr.sum()).item()

        message = (
            f"{len(null_counts)} column(s) contain null values, "
            f"affecting {affected_rows} row(s): {null_counts}."
        )

        return ValidationResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            category=self.category,
            severity=self.default_severity,
            status=Status.WARN,
            message=message,
            details={"null_counts": null_counts},
            affected_rows=affected_rows,
        )
