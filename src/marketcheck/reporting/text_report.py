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

# Raw ANSI rather than typer/click styling helpers, so this module keeps importing
# nothing but stdlib and models. Reporting is a presentation layer, not a CLI
# concern: rendering must stay callable (and assertable) without a CLI framework.
_RESET = "\x1b[0m"
_BOLD = "\x1b[1m"
_DIM = "\x1b[2m"

# A skip is deliberately given no hue -- only dimming. Colouring it would imply a
# verdict, and the whole point of SKIP is that no verdict was reached.
_STATUS_COLOR = {
    Status.PASS: "\x1b[32m",  # green
    Status.WARN: "\x1b[33m",  # yellow
    Status.FAIL: "\x1b[31m",  # red
    Status.SKIP: _DIM,
}


def _paint(text: str, status: Status, *, color: bool, bold: bool = False) -> str:
    """Wrap *text* in the ANSI colour for *status*, or return it unchanged.

    Args:
        text: Already-formatted text. Callers must apply any width padding
            *before* painting, because ANSI codes add characters that would
            otherwise be counted by ``str.format`` padding and break alignment.
        status: Status whose colour to apply.
        color: When False, returns *text* untouched.
        bold: Also embolden.

    Returns:
        The text, optionally wrapped in ANSI escape codes.
    """
    if not color:
        return text
    prefix = _STATUS_COLOR.get(status, "")
    if bold:
        prefix = _BOLD + prefix
    return f"{prefix}{text}{_RESET}" if prefix else text


def render_text(summary: DatasetSummary, *, color: bool = False) -> str:
    """Render a DatasetSummary as a human-readable text report.

    Args:
        summary: The aggregated validation summary.
        color: Emit ANSI colour codes. Defaults to False so the rendered string
            stays a plain, durable artifact -- callers writing to a file or
            comparing output get no escape codes unless they ask for them.

    Returns:
        A formatted multi-line string.
    """
    lines: list[str] = []

    # Header
    icon = _STATUS_ICON.get(summary.overall_status, "?")
    verdict = _paint(
        f"{icon}  {summary.overall_status.value.upper()}",
        summary.overall_status,
        color=color,
        bold=True,
    )
    lines.append(f"{'=' * 72}")
    lines.append(f"  MarketCheck Report  {verdict}")
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
            # Pad before painting so alignment is unaffected by escape codes.
            label = f"{icon} [{result.status.value.upper():4s}]"
            lines.append(
                f"  {_paint(label, result.status, color=color)} "
                f"{result.rule_name} ({result.rule_id})"
            )
            lines.append(f"         {result.message}")
            if result.affected_rows > 0:
                lines.append(f"         Affected rows: {result.affected_rows:,}")
            lines.append("")

    lines.append(f"{'=' * 72}")
    return "\n".join(lines)
