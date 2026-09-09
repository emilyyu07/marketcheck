"""
Plain-text report renderer.
Takes a DatasetSummary and produces a human-readable text report in string format
"""

from __future__ import annotations

from marketcheck.models.enums import Status
from marketcheck.models.result import DatasetSummary

_STATUS_ICON = {
    Status.PASS: "✓",
    Status.WARN: "⚠",
    Status.FAIL: "✗",
    # Visually neutral on purpose: a skip is neither success nor failure.
    Status.SKIP: "–",
}


def render_text(summary: DatasetSummary) -> str:
    """Render a DatasetSummary as a human-readable text report.

    Args:
        summary: The aggregated validation summary.

    Returns:
        A formatted multi-line string.
    """
    lines: list[str] = []

    # Header
    icon = _STATUS_ICON.get(summary.overall_status, "?")
    lines.append(f"{'=' * 72}")
    lines.append(f"  MarketCheck Report  {icon}  {summary.overall_status.value.upper()}")
    lines.append(f"{'=' * 72}")
    lines.append(f"  File:   {summary.source_path}")
    if summary.ticker:
        lines.append(f"  Ticker: {summary.ticker}")
    lines.append(f"  Rows:   {summary.row_count:,}")
    if summary.overall_message:
        lines.append(f"  Note:   {summary.overall_message}")
    if summary.start_time:
        lines.append(f"  Range:  {summary.start_time} → {summary.end_time}")
    lines.append("")

    # Counts
    lines.append(
        f"  Rules run: {summary.total_rules_run}  |  "
        f"Passed: {summary.total_passed}  |  "
        f"Warned: {summary.total_warned}  |  "
        f"Failed: {summary.total_failed}  |  "
        f"Skipped: {summary.total_skipped}"
    )
    lines.append(f"{'-' * 72}")

    # Individual results
    if not summary.results:
        lines.append("  No validation results.")
    else:
        for result in summary.results:
            icon = _STATUS_ICON.get(result.status, "?")
            lines.append(
                f"  {icon} [{result.status.value.upper():4s}] "
                f"{result.rule_name} ({result.rule_id})"
            )
            lines.append(f"         {result.message}")
            if result.affected_rows > 0:
                lines.append(f"         Affected rows: {result.affected_rows:,}")
            lines.append("")

    lines.append(f"{'=' * 72}")
    return "\n".join(lines)
