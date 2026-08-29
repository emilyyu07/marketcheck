"""Tests for temporal validation rules — stub phase."""

import pytest

from marketcheck.models.enums import Category
from marketcheck.validators import REGISTRY
from marketcheck.validators.temporal import (
    GapsWithinSession,
    MissingSessions,
    OutsideTradingHours,
    TimezoneInconsistency,
)

TEMPORAL_RULES = [
    MissingSessions,
    GapsWithinSession,
    OutsideTradingHours,
    TimezoneInconsistency,
]


class TestTemporalRulesRegistered:
    @pytest.mark.parametrize("rule_cls", TEMPORAL_RULES)
    def test_rule_is_registered(self, rule_cls: type) -> None:
        assert rule_cls in REGISTRY

    @pytest.mark.parametrize("rule_cls", TEMPORAL_RULES)
    def test_rule_has_correct_category(self, rule_cls: type) -> None:
        rule = rule_cls()
        assert rule.category == Category.TEMPORAL

    @pytest.mark.parametrize("rule_cls", TEMPORAL_RULES)
    def test_rule_raises_not_implemented(self, rule_cls: type) -> None:
        rule = rule_cls()
        with pytest.raises(NotImplementedError):
            rule.validate(None, None)  # type: ignore[arg-type]
