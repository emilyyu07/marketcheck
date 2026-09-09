"""Aggregator: combines individual rule results into a DatasetSummary (result summary)"""

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
    total_skipped = sum(1 for r in results if r.status == Status.SKIP)

    # Determine worst-case overall status.
    #
    # SKIP deliberately does not worsen the outcome -- a rule that could not run
    # found no problem. But it must not manufacture a clean bill of health either:
    # if NOTHING produced a real verdict, "pass" would be a false all-clear on a
    # file that was never actually checked. That case reports WARN instead.
    overall_message = ""
    if total_failed > 0:
        overall = Status.FAIL
    elif total_warned > 0:
        overall = Status.WARN
    elif total_passed > 0:
        overall = Status.PASS
    elif total_skipped > 0:
        overall = Status.WARN
        overall_message = (
            "No rule produced a verdict; every rule was skipped, so nothing about "
            "this data has actually been verified."
        )
    else:
        # No rules at all (e.g. every rule filtered out by config).
        overall = Status.PASS

    # A dataset with no rows cannot earn a clean bill of health. Some rules still
    # reach real verdicts on an empty frame (the columns exist, so the schema is
    # genuinely checkable), which would otherwise add up to an overall PASS on a
    # file containing no data at all.
    if dataset.row_count == 0 and overall != Status.FAIL:
        overall = Status.WARN
        overall_message = (
            "Dataset contains no rows, so no data could be validated."
        )

    return DatasetSummary(
        source_path=dataset.source_path,
        ticker=dataset.ticker,
        row_count=dataset.row_count,
        start_time=dataset.start_time,
        end_time=dataset.end_time,
        overall_status=overall,
        overall_message=overall_message,
        # "Attempted", not "produced a verdict", so that
        # passed + warned + failed + skipped reconciles to this exactly.
        total_rules_run=len(results),
        total_passed=total_passed,
        total_warned=total_warned,
        total_failed=total_failed,
        total_skipped=total_skipped,
        results=results,
    )
