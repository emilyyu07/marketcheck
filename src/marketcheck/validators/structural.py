"""Structural validation rules (5 rules)."""

from marketcheck.ingestion.schema import REQUIRED_COLUMNS
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
Rule 1: MissingColumns
This rule checks that all required columns are present in the dataset.
If any required columns are missing, the rule fails and reports which columns are missing.
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


@register
class InvalidDtypes(ValidationRule):
    rule_id = "structural.invalid_dtypes"
    rule_name = "Invalid Data Types"
    category = Category.STRUCTURAL
    default_severity = Severity.CRITICAL

    def validate(self, dataset: CanonicalDataset, context: RuleContext) -> ValidationResult:
        raise NotImplementedError("TODO: implement structural.invalid_dtypes")


@register
class UnsortedTimestamps(ValidationRule):
    rule_id = "structural.unsorted_timestamps"
    rule_name = "Unsorted Timestamps"
    category = Category.STRUCTURAL
    default_severity = Severity.WARNING

    def validate(self, dataset: CanonicalDataset, context: RuleContext) -> ValidationResult:
        raise NotImplementedError("TODO: implement structural.unsorted_timestamps")


@register
class NullValues(ValidationRule):
    rule_id = "structural.null_values"
    rule_name = "Null / Missing Values"
    category = Category.STRUCTURAL
    default_severity = Severity.WARNING

    def validate(self, dataset: CanonicalDataset, context: RuleContext) -> ValidationResult:
        raise NotImplementedError("TODO: implement structural.null_values")
