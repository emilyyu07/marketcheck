"""Structural validation rules (5 rules)."""

from marketcheck.models.dataset import CanonicalDataset
from marketcheck.models.enums import Category, Severity
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


@register
class MissingColumns(ValidationRule):
    rule_id = "structural.missing_columns"
    rule_name = "Missing Required Columns"
    category = Category.STRUCTURAL
    default_severity = Severity.CRITICAL

    def validate(self, dataset: CanonicalDataset, context: RuleContext) -> ValidationResult:
        raise NotImplementedError("TODO: implement structural.missing_columns")


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
