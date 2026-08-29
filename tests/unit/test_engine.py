"""Tests for the validation engine (runner + aggregator)."""

import polars as pl

from marketcheck.engine.aggregator import aggregate
from marketcheck.engine.runner import run_validation
from marketcheck.ingestion.canonicalize import to_canonical
from marketcheck.models.enums import Status


class TestRunner:
    def test_run_validation_with_stubs(self, sample_ohlcv_df: pl.DataFrame) -> None:
        """All rules are stubbed — runner should skip them and return results."""
        dataset = to_canonical(sample_ohlcv_df, source_path="test.csv")
        results = run_validation(dataset)
        # Should have one result per registered rule (all PASS placeholders)
        assert len(results) > 0
        # None should have FAIL status (since stubs are caught as NotImplementedError)
        assert all(r.status == Status.PASS for r in results)


class TestAggregator:
    def test_aggregate_empty(self, sample_ohlcv_df: pl.DataFrame) -> None:
        dataset = to_canonical(sample_ohlcv_df, source_path="test.csv")
        summary = aggregate([], dataset)
        assert summary.overall_status == Status.PASS
        assert summary.total_rules_run == 0

    def test_aggregate_with_stub_results(self, sample_ohlcv_df: pl.DataFrame) -> None:
        dataset = to_canonical(sample_ohlcv_df, source_path="test.csv")
        results = run_validation(dataset)
        summary = aggregate(results, dataset)
        assert summary.total_rules_run == len(results)
        assert summary.row_count == 10
