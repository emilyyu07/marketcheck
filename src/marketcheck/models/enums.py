"""Core enumerations used across MarketCheck."""

from enum import Enum


class Severity(str, Enum):
    """
    How severe a validation finding is.

    Severity belongs to a rule and is static. It describes how bad a finding would be
    if it occurred (critical indicates data in unusable, warning indicates data is 
    usable but has potential issues, info indicates data is usable and the finding is 
    informational).
    """

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class Status(str, Enum):
    """Outcome status for a single validation rule."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class Category(str, Enum):
    """
    
    Classification category for validation rules (what type of data they validate, i.e. 
    shape/columns, time, numbers, market specific concepts).
    
    """

    STRUCTURAL = "structural"
    TEMPORAL = "temporal"
    NUMERICAL = "numerical"
    FINANCIAL = "financial"
