"""Tests for Pydantic models and enums."""

import polars as pl

from marketcheck.models.config import ValidationConfig
from marketcheck.models.dataset import CanonicalDataset
from marketcheck.models.enums import Category, Severity, Status
from marketcheck.models.result import DatasetSummary, ValidationResult


class TestEnums:
    def test_severity_values(self) -> None:
        assert Severity.INFO == "info"
        assert Severity.WARNING == "warning"
        assert Severity.CRITICAL == "critical"

    def test_status_values(self) -> None:
        assert Status.PASS == "pass"
        assert Status.WARN == "warn"
        assert Status.FAIL == "fail"

    def test_category_values(self) -> None:
        assert Category.STRUCTURAL == "structural"
        assert Category.TEMPORAL == "temporal"
        assert Category.NUMERICAL == "numerical"
        assert Category.FINANCIAL == "financial"


class TestValidationConfig:
    def test_defaults(self) -> None:
        config = ValidationConfig()
        assert config.strict is False
        assert config.output_format == "text"


class TestCanonicalDataset:
    def test_creation(self) -> None:
        df = pl.DataFrame({"a": [1]})
        dataset = CanonicalDataset(df=df, source_path="test.csv")
        assert dataset.row_count == 0  # default, not auto-computed
        assert dataset.source_path == "test.csv"


class TestValidationResult:
    def test_creation(self) -> None:
        result = ValidationResult(
            rule_id="test.rule",
            rule_name="Test Rule",
            category=Category.STRUCTURAL,
            severity=Severity.INFO,
            status=Status.PASS,
            message="All good.",
        )
        assert result.rule_id == "test.rule"
        assert result.affected_rows == 0


class TestDatasetSummary:
    def test_defaults(self) -> None:
        summary = DatasetSummary(source_path="test.csv")
        assert summary.overall_status == Status.PASS
        assert summary.total_rules_run == 0
