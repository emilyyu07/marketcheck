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
    """Outcome status for a single validation rule.

    SKIP is distinct from PASS and the distinction is the point: PASS asserts the
    rule checked the data and found nothing wrong, while SKIP records that the
    rule could not evaluate the data at all. Reporting "pass" for a check that
    never ran is exactly the false assurance this tool exists to prevent.

    A rule skips for two reasons:
      1. It is not yet implemented (the runner catches NotImplementedError).
      2. A prerequisite is absent, so there is nothing to judge -- no `volume`
         column for VolumeAnomaly, no `timestamp` for the temporal rules, or a
         bar frequency that cannot be inferred credibly.

    SKIP never worsens the overall status: a skipped rule found no problem, it
    simply found no answer. But it must never be counted as a pass either.
    """

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


class Category(str, Enum):
    """
    
    Classification category for validation rules (what type of data they validate, i.e. 
    shape/columns, time, numbers, market specific concepts).
    
    """

    STRUCTURAL = "structural"
    TEMPORAL = "temporal"
    NUMERICAL = "numerical"
    FINANCIAL = "financial"
