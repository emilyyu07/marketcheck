"""Core enumerations used across MarketCheck."""

from enum import Enum


class Severity(str, Enum):
    """How severe a validation finding is."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class Status(str, Enum):
    """Outcome status for a single validation rule."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class Category(str, Enum):
    """Classification category for validation rules."""

    STRUCTURAL = "structural"
    TEMPORAL = "temporal"
    NUMERICAL = "numerical"
    FINANCIAL = "financial"
