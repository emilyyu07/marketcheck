"""Temporal validation rules (4 rules)."""

from marketcheck.models.dataset import CanonicalDataset
from marketcheck.models.enums import Category, Severity
from marketcheck.models.result import ValidationResult
from marketcheck.validators.base import RuleContext, ValidationRule, register


@register
class MissingSessions(ValidationRule):
    rule_id = "temporal.missing_sessions"
    rule_name = "Missing Trading Sessions"
    category = Category.TEMPORAL
    default_severity = Severity.WARNING

    def validate(self, dataset: CanonicalDataset, context: RuleContext) -> ValidationResult:
        raise NotImplementedError("TODO: implement temporal.missing_sessions")


@register
class GapsWithinSession(ValidationRule):
    rule_id = "temporal.gaps_within_session"
    rule_name = "Gaps Within Session"
    category = Category.TEMPORAL
    default_severity = Severity.WARNING

    def validate(self, dataset: CanonicalDataset, context: RuleContext) -> ValidationResult:
        raise NotImplementedError("TODO: implement temporal.gaps_within_session")


@register
class OutsideTradingHours(ValidationRule):
    rule_id = "temporal.outside_trading_hours"
    rule_name = "Data Outside Trading Hours"
    category = Category.TEMPORAL
    default_severity = Severity.INFO

    def validate(self, dataset: CanonicalDataset, context: RuleContext) -> ValidationResult:
        raise NotImplementedError("TODO: implement temporal.outside_trading_hours")


@register
class TimezoneInconsistency(ValidationRule):
    rule_id = "temporal.timezone_inconsistency"
    rule_name = "Timezone Inconsistency"
    category = Category.TEMPORAL
    default_severity = Severity.CRITICAL

    def validate(self, dataset: CanonicalDataset, context: RuleContext) -> ValidationResult:
        raise NotImplementedError("TODO: implement temporal.timezone_inconsistency")
