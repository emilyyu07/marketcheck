"""Validation result and summary models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from marketcheck.models.enums import Category, Severity, Status


class TimeRange(BaseModel):
    """A time range within the dataset."""

    start: datetime
    end: datetime


class ValidationResult(BaseModel):
    """The outcome of a single validation rule applied to a dataset."""

    rule_id: str
    """Unique identifier for the rule (e.g. 'structural.duplicate_timestamps')."""

    rule_name: str
    """Human-readable name of the rule."""

    category: Category
    """Which category this rule belongs to."""

    severity: Severity
    """The severity level of this rule's findings."""

    status: Status
    """Whether the check passed, warned, or failed."""

    message: str
    """Human-readable explanation of the finding."""

    details: dict[str, object] = {}
    """Structured details (row indices, example values, etc.)."""

    affected_rows: int = 0
    """Number of rows affected by this finding."""

    affected_ranges: list[TimeRange] = []
    """Time ranges where issues were found."""


class DatasetSummary(BaseModel):
    """Aggregated summary of all validation results for a dataset."""

    source_path: str
    """Path to the validated file."""

    ticker: str = ""
    """Ticker symbol, if known."""

    row_count: int = 0
    """Total rows in the dataset."""

    start_time: datetime | None = None
    """Earliest timestamp."""

    end_time: datetime | None = None
    """Latest timestamp."""

    overall_status: Status = Status.PASS
    """Worst-case status across all rules."""

    total_rules_run: int = 0
    """Number of validation rules that were executed."""

    total_passed: int = 0
    total_warned: int = 0
    total_failed: int = 0
    total_skipped: int = 0

    results: list[ValidationResult] = []
    """Individual rule results."""
