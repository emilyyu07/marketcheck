"""Base validation rule interface, registry, and decorator (core data contracts)"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

from marketcheck.calendar.sessions import MarketCalendar
from marketcheck.models.config import ValidationConfig
from marketcheck.models.dataset import CanonicalDataset
from marketcheck.models.enums import Category, Severity
from marketcheck.models.result import ValidationResult


@dataclass
class RuleContext:
    """Contextual information passed to each rule during validation.

    Constructed once per ``run_validation()`` call and shared across every
    rule in that run -- so ``calendar`` is built a single time per
    validation run, not once per rule that needs it.
    """

    config: ValidationConfig = field(default_factory=ValidationConfig)
    calendar: MarketCalendar = field(default_factory=MarketCalendar)


class ValidationRule(abc.ABC):
    """Abstract base class for all validation rules.

    Each rule must declare its identity (rule_id, category, default_severity)
    and implement the ``validate`` method.
    """

    rule_id: str
    """Unique dot-separated identifier, e.g. 'structural.duplicate_timestamps'."""

    rule_name: str
    """Human-readable name shown in reports."""

    category: Category
    """Which category this rule belongs to."""

    default_severity: Severity
    """Default severity; can be overridden via config."""

    @abc.abstractmethod
    def validate(
        self, dataset: CanonicalDataset, context: RuleContext
    ) -> ValidationResult:
        """Run this check against *dataset* and return a result.

        Args:
            dataset: The canonical dataset to validate.
            context: Runtime context (config, calendar, etc.).

        Returns:
            A ValidationResult describing the outcome.
        """
        ...


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

REGISTRY: list[type[ValidationRule]] = []
"""All registered validation rule classes."""


def register(cls: type[ValidationRule]) -> type[ValidationRule]:
    """Class decorator that adds a ValidationRule subclass to the registry."""
    REGISTRY.append(cls)
    return cls
