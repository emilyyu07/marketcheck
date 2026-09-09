"""Validation runner — iterates registered rules and collects results (rule execution)"""

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

    Iterates through all registered validation rules, applies each to the provided
    dataset, and collects one result per rule. A rule that cannot run is recorded
    as SKIP rather than omitted or reported as a pass.

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

        # A rule the user disabled by name is still REPORTED, as a skip. Omitting
        # it entirely would leave a reader unable to tell "14 rules, 3 disabled"
        # from "this tool only has 11 rules" -- the same opacity as reporting an
        # unimplemented rule as a pass.
        if rule.rule_id in config.disabled_rules:
            logger.debug("Skipping disabled rule: %s", rule.rule_id)
            results.append(
                ValidationResult(
                    rule_id=rule.rule_id,
                    rule_name=rule.rule_name,
                    category=rule.category,
                    severity=rule.default_severity,
                    status=Status.SKIP,
                    message="Rule disabled by configuration.",
                )
            )
            continue

        # A category filter, by contrast, is an explicit narrowing of scope: the
        # user asked for one category, so listing every other rule as skipped
        # would be noise rather than transparency. Those are omitted.
        if config.enabled_categories and rule.category.value not in config.enabled_categories:
            logger.debug("Skipping rule outside enabled categories: %s", rule.rule_id)
            continue

        try:
            result = rule.validate(dataset, context)
            results.append(result)
        except NotImplementedError as exc:
            logger.warning("Rule not yet implemented, skipping: %s (%s)", rule.rule_id, exc)
            # Record the rule as SKIPPED, never as passed. A PASS placeholder here
            # would inflate total_passed and tell the user an unimplemented check
            # had verified their data.
            results.append(
                ValidationResult(
                    rule_id=rule.rule_id,
                    rule_name=rule.rule_name,
                    category=rule.category,
                    severity=rule.default_severity,
                    status=Status.SKIP,
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
