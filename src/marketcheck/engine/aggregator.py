"""Aggregator — combines individual rule results into a DatasetSummary."""

from __future__ import annotations

from marketcheck.models.dataset import CanonicalDataset
from marketcheck.models.enums import Status
from marketcheck.models.result import DatasetSummary, ValidationResult


def aggregate(
    results: list[ValidationResult],
    dataset: CanonicalDataset,
) -> DatasetSummary:
    """Aggregate a list of validation results into a single summary.

    Args:
        results: Individual rule results.
        dataset: The dataset that was validated (for metadata).

    Returns:
        A DatasetSummary with counts, overall status, and all results.
    """
    total_passed = sum(1 for r in results if r.status == Status.PASS)
    total_warned = sum(1 for r in results if r.status == Status.WARN)
    total_failed = sum(1 for r in results if r.status == Status.FAIL)

    # Determine worst-case overall status
    if total_failed > 0:
        overall = Status.FAIL
    elif total_warned > 0:
        overall = Status.WARN
    else:
        overall = Status.PASS

    return DatasetSummary(
        source_path=dataset.source_path,
        ticker=dataset.ticker,
        row_count=dataset.row_count,
        start_time=dataset.start_time,
        end_time=dataset.end_time,
        overall_status=overall,
        total_rules_run=len(results),
        total_passed=total_passed,
        total_warned=total_warned,
        total_failed=total_failed,
        total_skipped=0,
        results=results,
    )
