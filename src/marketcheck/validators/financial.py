"""Financial validation rules (1 rule)."""

from marketcheck.models.dataset import CanonicalDataset
from marketcheck.models.enums import Category, Severity
from marketcheck.models.result import ValidationResult
from marketcheck.validators.base import RuleContext, ValidationRule, register


@register
class CorporateActionDiscontinuity(ValidationRule):
    rule_id = "financial.corporate_action_discontinuity"
    rule_name = "Corporate Action Discontinuity"
    category = Category.FINANCIAL
    default_severity = Severity.INFO

    def validate(self, dataset: CanonicalDataset, context: RuleContext) -> ValidationResult:
        raise NotImplementedError("TODO: implement financial.corporate_action_discontinuity")
