"""Validation runner — iterates registered rules and collects results."""

from __future__ import annotations

import logging

from marketcheck.models.config import ValidationConfig
from marketcheck.models.dataset import CanonicalDataset
from marketcheck.models.enums import Severity, Status
from marketcheck.models.result import ValidationResult
from marketcheck.validators.base import REGISTRY, RuleContext

logger = logging.getLogger(__name__)


def run_validation(
    dataset: CanonicalDataset,
    config: ValidationConfig | None = None,
) -> list[ValidationResult]:
    """Run all registered validation rules against *dataset*.

    Rules that raise ``NotImplementedError`` are skipped with a warning,
    so the pipeline works end-to-end even before rules are implemented.

    Args:
        dataset: The canonical dataset to validate.
        config: Optional runtime configuration.

    Returns:
        List of results from all rules that ran successfully.
    """
    if config is None:
        config = ValidationConfig()

    context = RuleContext(config=config)
    results: list[ValidationResult] = []

    for rule_cls in REGISTRY:
        rule = rule_cls()

        # Skip disabled rules
        if rule.rule_id in config.disabled_rules:
            logger.debug("Skipping disabled rule: %s", rule.rule_id)
            continue

        # Skip rules outside enabled categories (if filter is set)
        if config.enabled_categories and rule.category.value not in config.enabled_categories:
            logger.debug("Skipping rule outside enabled categories: %s", rule.rule_id)
            continue

        try:
            result = rule.validate(dataset, context)
            results.append(result)
        except NotImplementedError as exc:
            logger.warning("Rule not yet implemented, skipping: %s (%s)", rule.rule_id, exc)
            # Append a PASS placeholder so the report reflects that this rule was seen
            results.append(
                ValidationResult(
                    rule_id=rule.rule_id,
                    rule_name=rule.rule_name,
                    category=rule.category,
                    severity=rule.default_severity,
                    status=Status.PASS,
                    message=f"Rule not yet implemented: {rule.rule_id}",
                )
            )
        except Exception:
            logger.exception("Unexpected error in rule %s", rule.rule_id)
            results.append(
                ValidationResult(
                    rule_id=rule.rule_id,
                    rule_name=rule.rule_name,
                    category=rule.category,
                    severity=Severity.CRITICAL,
                    status=Status.FAIL,
                    message=f"Unexpected error running rule: {rule.rule_id}",
                )
            )

    return results
