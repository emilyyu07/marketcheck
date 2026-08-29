"""Validation configuration model."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from marketcheck.models.enums import Severity


class ValidationConfig(BaseModel):
    """Runtime configuration for a validation run."""

    strict: bool = False
    """If True, treat WARNINGs as FAILs."""

    enabled_categories: list[str] = []
    """If non-empty, only run rules in these categories. Empty = run all."""

    disabled_rules: list[str] = []
    """Rule IDs to skip."""

    severity_overrides: dict[str, Severity] = {}
    """Override default severity for specific rule IDs."""

    max_rows_in_details: int = 50
    """Cap on how many affected rows to include in detailed output."""

    output_format: str = "text"
    """Default output format: 'text' or 'json'."""

    output_path: Path | None = None
    """If set, write the report to this path instead of stdout."""
