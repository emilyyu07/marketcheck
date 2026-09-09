"""Tests for the validation engine (runner + aggregator)."""

from __future__ import annotations

from collections.abc import Iterator

import polars as pl
import pytest

from marketcheck.engine.aggregator import aggregate
from marketcheck.engine.runner import run_validation
from marketcheck.ingestion.canonicalize import to_canonical
from marketcheck.models.config import ValidationConfig
from marketcheck.models.dataset import CanonicalDataset
from marketcheck.models.enums import Category, Severity, Status
from marketcheck.models.result import ValidationResult
from marketcheck.validators.base import REGISTRY, RuleContext, ValidationRule


def _result(status: Status, rule_id: str = "test.rule") -> ValidationResult:
    """A minimal ValidationResult with the given status."""
    return ValidationResult(
        rule_id=rule_id,
        rule_name="Test Rule",
        category=Category.STRUCTURAL,
        severity=Severity.WARNING,
        status=status,
        message="test",
    )


@pytest.fixture
def broken_rule() -> Iterator[type[ValidationRule]]:
    """Temporarily register a rule that raises, to exercise the runner's
    generic-exception path. Removed afterwards so the global REGISTRY is
    unchanged for other tests."""

    class DeliberatelyBrokenRule(ValidationRule):
        rule_id = "test.broken"
        rule_name = "Deliberately Broken Rule"
        category = Category.STRUCTURAL
        default_severity = Severity.INFO

        def validate(
            self, dataset: CanonicalDataset, context: RuleContext
        ) -> ValidationResult:
            raise ValueError("boom")

    REGISTRY.append(DeliberatelyBrokenRule)
    try:
        yield DeliberatelyBrokenRule
    finally:
        REGISTRY.remove(DeliberatelyBrokenRule)


class TestRunner:
    def test_runs_one_rule_per_registered_rule(self, sample_ohlcv_df: pl.DataFrame) -> None:
        dataset = to_canonical(sample_ohlcv_df, source_path="test.csv")
        results = run_validation(dataset)

        assert len(results) == len(REGISTRY)
        assert len({r.rule_id for r in results}) == len(results)

    def test_unimplemented_rule_is_skipped_not_passed(
        self, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        """A stub must never be reported as a pass — that would claim an
        unimplemented check had verified the data."""
        dataset = to_canonical(sample_ohlcv_df, source_path="test.csv")
        results = run_validation(dataset)

        stubs = [r for r in results if "not yet implemented" in r.message]
        assert stubs, "expected at least one unimplemented rule"
        for stub in stubs:
            assert stub.status == Status.SKIP

    def test_clean_data_produces_no_failures(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """The shared fixture is deliberately clean, so no rule should fail on it."""
        dataset = to_canonical(sample_ohlcv_df, source_path="test.csv")
        results = run_validation(dataset)

        assert [r.rule_id for r in results if r.status == Status.FAIL] == []

    def test_disabled_rule_does_not_evaluate(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """A disabled rule must not run. It is still listed (as a skip) -- see
        TestRunnerSkipVisibility -- but it must not produce a verdict."""
        dataset = to_canonical(sample_ohlcv_df, source_path="test.csv")
        config = ValidationConfig(disabled_rules=["structural.missing_columns"])
        results = run_validation(dataset, config=config)

        disabled = next(r for r in results if r.rule_id == "structural.missing_columns")
        assert disabled.status == Status.SKIP

    def test_enabled_categories_filters_rules(self, sample_ohlcv_df: pl.DataFrame) -> None:
        dataset = to_canonical(sample_ohlcv_df, source_path="test.csv")
        config = ValidationConfig(enabled_categories=["structural"])
        results = run_validation(dataset, config=config)

        assert results
        assert {r.category for r in results} == {Category.STRUCTURAL}

    def test_unexpected_exception_becomes_critical_failure(
        self, sample_ohlcv_df: pl.DataFrame, broken_rule: type[ValidationRule]
    ) -> None:
        """A crashing rule must surface as a FAIL, not vanish from the report."""
        dataset = to_canonical(sample_ohlcv_df, source_path="test.csv")
        results = run_validation(dataset)

        broken = [r for r in results if r.rule_id == "test.broken"]
        assert len(broken) == 1
        assert broken[0].status == Status.FAIL
        assert broken[0].severity == Severity.CRITICAL

    def test_crashing_rule_does_not_stop_other_rules(
        self, sample_ohlcv_df: pl.DataFrame, broken_rule: type[ValidationRule]
    ) -> None:
        dataset = to_canonical(sample_ohlcv_df, source_path="test.csv")
        results = run_validation(dataset)

        assert len(results) == len(REGISTRY)


class TestAggregatorCounts:
    def test_empty_results(self, sample_ohlcv_df: pl.DataFrame) -> None:
        dataset = to_canonical(sample_ohlcv_df, source_path="test.csv")
        summary = aggregate([], dataset)

        assert summary.overall_status == Status.PASS
        assert summary.total_rules_run == 0

    def test_skips_are_counted_separately(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """The core fix: a skip must not be counted as a pass."""
        dataset = to_canonical(sample_ohlcv_df, source_path="test.csv")
        summary = aggregate(
            [_result(Status.PASS, "a"), _result(Status.SKIP, "b"), _result(Status.SKIP, "c")],
            dataset,
        )

        assert summary.total_passed == 1
        assert summary.total_skipped == 2

    def test_counters_reconcile_to_rules_run(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """total_rules_run means 'attempted', so the four counters sum to it
        exactly — otherwise the report would not be self-consistent."""
        dataset = to_canonical(sample_ohlcv_df, source_path="test.csv")
        results = [
            _result(Status.PASS, "a"),
            _result(Status.WARN, "b"),
            _result(Status.FAIL, "c"),
            _result(Status.SKIP, "d"),
        ]
        summary = aggregate(results, dataset)

        total = (
            summary.total_passed
            + summary.total_warned
            + summary.total_failed
            + summary.total_skipped
        )
        assert total == summary.total_rules_run == 4

    def test_real_run_counters_reconcile(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """Same invariant, end to end through the real registry."""
        dataset = to_canonical(sample_ohlcv_df, source_path="test.csv")
        summary = aggregate(run_validation(dataset), dataset)

        assert (
            summary.total_passed
            + summary.total_warned
            + summary.total_failed
            + summary.total_skipped
            == summary.total_rules_run
        )

    def test_skipped_count_is_no_longer_hardcoded_zero(
        self, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        """Regression guard: total_skipped was previously hardcoded to 0 while
        unimplemented rules were reported as passes."""
        dataset = to_canonical(sample_ohlcv_df, source_path="test.csv")
        summary = aggregate(run_validation(dataset), dataset)

        assert summary.total_skipped > 0


class TestAggregatorOverallStatus:
    """Precedence: FAIL > WARN > PASS, with SKIP never worsening the outcome."""

    def _summary(self, statuses: list[Status], df: pl.DataFrame) -> Status:
        dataset = to_canonical(df, source_path="test.csv")
        results = [_result(s, f"r{i}") for i, s in enumerate(statuses)]
        return aggregate(results, dataset).overall_status

    def test_fail_dominates(self, sample_ohlcv_df: pl.DataFrame) -> None:
        assert (
            self._summary(
                [Status.PASS, Status.WARN, Status.FAIL, Status.SKIP], sample_ohlcv_df
            )
            == Status.FAIL
        )

    def test_warn_when_no_failures(self, sample_ohlcv_df: pl.DataFrame) -> None:
        assert (
            self._summary([Status.PASS, Status.WARN, Status.SKIP], sample_ohlcv_df)
            == Status.WARN
        )

    def test_pass_when_only_passes_and_skips(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """A skip does not worsen the outcome: the rule found no problem, it
        simply found no answer."""
        assert (
            self._summary([Status.PASS, Status.SKIP, Status.SKIP], sample_ohlcv_df)
            == Status.PASS
        )

    def test_all_skipped_is_not_a_pass(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """DD4. If nothing produced a real verdict, 'pass' would be a false
        all-clear on a file that was never actually checked."""
        assert (
            self._summary([Status.SKIP, Status.SKIP], sample_ohlcv_df) == Status.WARN
        )


class TestRunnerSkipVisibility:
    """A rule that did not run must still be visible in the report."""

    def test_disabled_rule_is_reported_as_skipped(
        self, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        """Omitting it would leave a reader unable to tell "14 rules, 1 disabled"
        from "this tool only has 13 rules"."""
        dataset = to_canonical(sample_ohlcv_df, source_path="test.csv")
        config = ValidationConfig(disabled_rules=["structural.missing_columns"])
        results = run_validation(dataset, config=config)

        disabled = [r for r in results if r.rule_id == "structural.missing_columns"]
        assert len(disabled) == 1
        assert disabled[0].status == Status.SKIP
        assert "disabled" in disabled[0].message.lower()

    def test_disabled_rule_still_counted_in_rules_run(
        self, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        dataset = to_canonical(sample_ohlcv_df, source_path="test.csv")
        config = ValidationConfig(disabled_rules=["structural.missing_columns"])
        results = run_validation(dataset, config=config)

        assert len(results) == len(REGISTRY)

    def test_category_filter_omits_rather_than_lists_skips(
        self, sample_ohlcv_df: pl.DataFrame
    ) -> None:
        """A category filter is an explicit narrowing of scope, so listing every
        other rule as skipped would be noise rather than transparency."""
        dataset = to_canonical(sample_ohlcv_df, source_path="test.csv")
        config = ValidationConfig(enabled_categories=["structural"])
        results = run_validation(dataset, config=config)

        assert len(results) < len(REGISTRY)
        assert {r.category for r in results} == {Category.STRUCTURAL}


class TestZeroRowDataset:
    """A dataset with no rows must not earn a clean bill of health."""

    @staticmethod
    def _empty_dataset() -> CanonicalDataset:
        df = pl.DataFrame(
            {"timestamp": [], "open": [], "high": [], "low": [], "close": [], "volume": []},
            schema={
                "timestamp": pl.Datetime("us"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
            },
        )
        return to_canonical(df, source_path="empty.csv")

    def test_zero_rows_is_not_a_pass(self) -> None:
        """Some rules reach real verdicts on an empty frame (the columns exist, so
        the schema is genuinely checkable), which would otherwise add up to an
        overall PASS on a file containing no data at all."""
        dataset = self._empty_dataset()
        summary = aggregate(run_validation(dataset), dataset)

        assert summary.overall_status == Status.WARN

    def test_zero_rows_explains_itself(self) -> None:
        dataset = self._empty_dataset()
        summary = aggregate(run_validation(dataset), dataset)

        assert "no rows" in summary.overall_message.lower()

    def test_all_skipped_explains_itself(self, sample_ohlcv_df: pl.DataFrame) -> None:
        dataset = to_canonical(sample_ohlcv_df, source_path="test.csv")
        summary = aggregate([_result(Status.SKIP, "a")], dataset)

        assert summary.overall_status == Status.WARN
        assert "verified" in summary.overall_message.lower()

    def test_normal_run_has_no_overall_message(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """The note is only for outcomes that are not self-evident from the rules."""
        dataset = to_canonical(sample_ohlcv_df, source_path="test.csv")
        summary = aggregate([_result(Status.PASS, "a")], dataset)

        assert summary.overall_message == ""
