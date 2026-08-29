"""Numerical validation rules (3 rules)."""

from marketcheck.models.dataset import CanonicalDataset
from marketcheck.models.enums import Category, Severity
from marketcheck.models.result import ValidationResult
from marketcheck.validators.base import RuleContext, ValidationRule, register


@register
class OhlcRangeViolation(ValidationRule):
    rule_id = "numerical.ohlc_range_violation"
    rule_name = "OHLC Range Violation"
    category = Category.NUMERICAL
    default_severity = Severity.CRITICAL

    def validate(self, dataset: CanonicalDataset, context: RuleContext) -> ValidationResult:
        raise NotImplementedError("TODO: implement numerical.ohlc_range_violation")


@register
class VolumeAnomaly(ValidationRule):
    rule_id = "numerical.volume_anomaly"
    rule_name = "Volume Anomaly"
    category = Category.NUMERICAL
    default_severity = Severity.WARNING

    def validate(self, dataset: CanonicalDataset, context: RuleContext) -> ValidationResult:
        raise NotImplementedError("TODO: implement numerical.volume_anomaly")


@register
class SuspiciousPriceJump(ValidationRule):
    rule_id = "numerical.suspicious_price_jump"
    rule_name = "Suspicious Price Jump"
    category = Category.NUMERICAL
    default_severity = Severity.WARNING

    def validate(self, dataset: CanonicalDataset, context: RuleContext) -> ValidationResult:
        raise NotImplementedError("TODO: implement numerical.suspicious_price_jump")
