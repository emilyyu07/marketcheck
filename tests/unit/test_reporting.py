"""Tests for the report renderers (text + JSON)."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from marketcheck.models.enums import Category, Severity, Status
from marketcheck.models.result import DatasetSummary, ValidationResult
from marketcheck.reporting.json_report import render_json
from marketcheck.reporting.text_report import render_text


def _result(
    status: Status,
    rule_id: str = "structural.missing_columns",
    rule_name: str = "Missing Columns",
    message: str = "example message",
    affected_rows: int = 0,
    details: dict[str, object] | None = None,
) -> ValidationResult:
    return ValidationResult(
        rule_id=rule_id,
        rule_name=rule_name,
        category=Category.STRUCTURAL,
        severity=Severity.CRITICAL,
        status=status,
        message=message,
        affected_rows=affected_rows,
        details=details or {},
    )


def _summary(**overrides: object) -> DatasetSummary:
    base: dict[str, object] = {
        "source_path": "AAPL_1min.csv",
        "ticker": "AAPL",
        "row_count": 1234,
        "overall_status": Status.PASS,
        "total_rules_run": 1,
        "total_passed": 1,
        "results": [_result(Status.PASS)],
    }
    base.update(overrides)
    return DatasetSummary(**base)  # type: ignore[arg-type]


class TestRenderText:
    def test_includes_header_and_metadata(self) -> None:
        report = render_text(_summary())

        assert "MarketCheck Report" in report
        assert "AAPL_1min.csv" in report
        assert "AAPL" in report

    def test_row_count_is_thousands_separated(self) -> None:
        assert "1,234" in render_text(_summary())

    def test_shows_overall_status(self) -> None:
        report = render_text(_summary(overall_status=Status.FAIL))

        assert "FAIL" in report

    @pytest.mark.parametrize(
        ("status", "icon"),
        [(Status.PASS, "✓"), (Status.WARN, "⚠"), (Status.FAIL, "✗"), (Status.SKIP, "–")],
    )
    def test_each_status_has_its_own_icon(self, status: Status, icon: str) -> None:
        """SKIP's icon is visually neutral on purpose: a skip is neither success
        nor failure, and reusing the pass tick would restate the old defect."""
        report = render_text(_summary(results=[_result(status)], overall_status=status))

        assert icon in report

    def test_skip_is_labelled_distinctly_from_pass(self) -> None:
        report = render_text(_summary(results=[_result(Status.SKIP)]))

        assert "[SKIP]" in report
        assert "[PASS]" not in report

    def test_counts_line_includes_skipped(self) -> None:
        report = render_text(
            _summary(
                total_rules_run=4,
                total_passed=1,
                total_warned=1,
                total_failed=1,
                total_skipped=1,
            )
        )

        assert "Skipped: 1" in report
        assert "Rules run: 4" in report

    def test_overall_message_is_rendered(self) -> None:
        """A WARN caused by "nothing could be checked" must explain itself, or the
        reader sees a bare status with no reason."""
        report = render_text(
            _summary(
                overall_status=Status.WARN,
                overall_message="Dataset contains no rows, so no data could be validated.",
            )
        )

        assert "no rows" in report

    def test_no_note_line_when_message_absent(self) -> None:
        assert "Note:" not in render_text(_summary())

    def test_affected_rows_shown_only_when_nonzero(self) -> None:
        with_rows = render_text(
            _summary(results=[_result(Status.FAIL, affected_rows=42)])
        )
        without = render_text(_summary(results=[_result(Status.PASS, affected_rows=0)]))

        assert "42" in with_rows
        assert "Affected rows" not in without

    def test_rule_id_and_message_present(self) -> None:
        report = render_text(
            _summary(results=[_result(Status.FAIL, message="3 columns missing")])
        )

        assert "structural.missing_columns" in report
        assert "3 columns missing" in report

    def test_empty_results_is_stated_explicitly(self) -> None:
        report = render_text(_summary(results=[], total_rules_run=0, total_passed=0))

        assert "No validation results." in report

    def test_date_range_shown_when_known(self) -> None:
        report = render_text(
            _summary(
                start_time=datetime(2024, 1, 2, 9, 30),
                end_time=datetime(2024, 1, 2, 16, 0),
            )
        )

        assert "2024-01-02 09:30" in report

    def test_output_is_a_single_string(self) -> None:
        report = render_text(_summary())

        assert isinstance(report, str)
        assert report.count("\n") > 5


class TestRenderJson:
    def test_output_is_valid_json(self) -> None:
        data = json.loads(render_json(_summary()))

        assert data["overall_status"] == "pass"

    def test_includes_all_counters(self) -> None:
        data = json.loads(
            render_json(
                _summary(
                    total_rules_run=4,
                    total_passed=1,
                    total_warned=1,
                    total_failed=1,
                    total_skipped=1,
                )
            )
        )

        assert data["total_skipped"] == 1
        assert data["total_rules_run"] == 4

    def test_skip_status_serialised_as_skip(self) -> None:
        data = json.loads(render_json(_summary(results=[_result(Status.SKIP)])))

        assert data["results"][0]["status"] == "skip"

    def test_overall_message_included(self) -> None:
        data = json.loads(
            render_json(_summary(overall_message="Dataset contains no rows."))
        )

        assert data["overall_message"] == "Dataset contains no rows."

    def test_datetimes_are_serialised(self) -> None:
        data = json.loads(
            render_json(
                _summary(
                    start_time=datetime(2024, 1, 2, 9, 30),
                    end_time=datetime(2024, 1, 2, 16, 0),
                )
            )
        )

        assert data["start_time"].startswith("2024-01-02")

    def test_details_with_datetime_survive(self) -> None:
        """Rules put timestamps in details (CorporateActionDiscontinuity,
        GapsWithinSession), so this must not break serialisation."""
        data = json.loads(
            render_json(
                _summary(
                    results=[
                        _result(
                            Status.WARN,
                            details={
                                "violations": [
                                    {
                                        "index": 1,
                                        "gap_start": datetime(2024, 1, 2, 9, 33),
                                        "gap_duration": "4min",
                                    }
                                ]
                            },
                        )
                    ]
                )
            )
        )

        violation = data["results"][0]["details"]["violations"][0]
        assert violation["gap_start"].startswith("2024-01-02")
        assert violation["gap_duration"] == "4min"

    def test_enums_serialise_as_their_values(self) -> None:
        data = json.loads(render_json(_summary(results=[_result(Status.FAIL)])))
        result = data["results"][0]

        assert result["category"] == "structural"
        assert result["severity"] == "critical"

    def test_is_pretty_printed(self) -> None:
        """Indented output so a human can read the JSON directly."""
        assert "\n  " in render_json(_summary())

    def test_empty_results_serialise(self) -> None:
        data = json.loads(
            render_json(_summary(results=[], total_rules_run=0, total_passed=0))
        )

        assert data["results"] == []
