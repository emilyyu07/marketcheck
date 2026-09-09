"""End-to-end CLI integration tests."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from marketcheck.cli.main import (
    EXIT_OK,
    EXIT_TOOL_ERROR,
    EXIT_VALIDATION_FAILED,
    app,
)

runner = CliRunner()

_HEADER = "timestamp,open,high,low,close,volume\n"


def _write(path: Path, rows: str) -> Path:
    path.write_text(_HEADER + rows, encoding="utf-8")
    return path


def _clean_csv(tmp_path: Path) -> Path:
    """Two well-formed bars inside regular trading hours."""
    return _write(
        tmp_path / "clean.csv",
        "2024-01-02 09:30:00,100,101,99,100,1000\n"
        "2024-01-02 09:31:00,100,101,99,100,1000\n",
    )


def _warning_csv(tmp_path: Path) -> Path:
    """A bar at 20:00 trips OutsideTradingHours (WARN), nothing fails."""
    return _write(
        tmp_path / "warn.csv",
        "2024-01-02 09:30:00,100,101,99,100,1000\n"
        "2024-01-02 20:00:00,100,101,99,100,1000\n",
    )


def _failing_csv(tmp_path: Path) -> Path:
    """high < low plus a duplicate timestamp: two CRITICAL rules fail."""
    return _write(
        tmp_path / "fail.csv",
        "2024-01-02 09:30:00,100,90,110,100,1000\n"
        "2024-01-02 09:30:00,100,101,99,100,1000\n",
    )


class TestCliOutput:
    def test_text_format(self, sample_csv_path: Path) -> None:
        result = runner.invoke(app, [str(sample_csv_path)])

        assert result.exit_code == EXIT_OK
        assert "MarketCheck Report" in result.stdout

    def test_json_format(self, sample_csv_path: Path) -> None:
        result = runner.invoke(app, [str(sample_csv_path), "--format", "json"])

        assert result.exit_code == EXIT_OK
        data = json.loads(result.stdout)
        assert "overall_status" in data
        assert "results" in data

    def test_parquet_input(self, sample_parquet_path: Path) -> None:
        result = runner.invoke(app, [str(sample_parquet_path), "--format", "json"])

        assert result.exit_code == EXIT_OK

    def test_output_file(self, sample_csv_path: Path, tmp_path: Path) -> None:
        output_file = tmp_path / "report.txt"
        result = runner.invoke(app, [str(sample_csv_path), "--output", str(output_file)])

        assert result.exit_code == EXIT_OK
        assert output_file.exists()
        assert output_file.read_text()

    def test_skipped_rules_are_reported(self, sample_csv_path: Path) -> None:
        """Skips must be visible in the report rather than folded into passes."""
        result = runner.invoke(app, [str(sample_csv_path), "--format", "json"])
        data = json.loads(result.stdout)

        assert data["total_skipped"] > 0
        assert any(r["status"] == "skip" for r in data["results"])


class TestCliExitCodes:
    """The exit-code contract: 0 = clean, 1 = validation failed, 2 = tool error.

    Validation failure and tool failure are distinct so a CI pipeline can tell
    "the data is bad" from "the tool could not run".
    """

    def test_clean_data_exits_zero(self, tmp_path: Path) -> None:
        result = runner.invoke(app, [str(_clean_csv(tmp_path))])

        assert result.exit_code == EXIT_OK

    def test_failing_data_exits_nonzero(self, tmp_path: Path) -> None:
        """The defect this replaces: the CLI used to exit 0 on FAIL, which made
        it silently useless in CI."""
        path = _failing_csv(tmp_path)
        json_result = runner.invoke(app, [str(path), "--format", "json"])
        assert json.loads(json_result.stdout)["overall_status"] == "fail"

        result = runner.invoke(app, [str(path)])
        assert result.exit_code == EXIT_VALIDATION_FAILED

    def test_warnings_do_not_fail_by_default(self, tmp_path: Path) -> None:
        """Warnings cover findings a human must judge — a genuine trading halt, a
        real 70% biotech drop — so failing the build on them by default would make
        the tool unusable on real data."""
        path = _warning_csv(tmp_path)
        json_result = runner.invoke(app, [str(path), "--format", "json"])
        assert json.loads(json_result.stdout)["overall_status"] == "warn"

        result = runner.invoke(app, [str(path)])
        assert result.exit_code == EXIT_OK

    def test_missing_file_is_a_tool_error(self, tmp_path: Path) -> None:
        result = runner.invoke(app, [str(tmp_path / "nonexistent.csv")])

        assert result.exit_code == EXIT_TOOL_ERROR

    def test_unsupported_extension_is_a_tool_error(self, tmp_path: Path) -> None:
        path = tmp_path / "data.xyz"
        path.write_text("nope", encoding="utf-8")
        result = runner.invoke(app, [str(path)])

        assert result.exit_code == EXIT_TOOL_ERROR


class TestCliStrict:
    """--strict must actually change behaviour; it was previously a silent no-op."""

    def test_strict_fails_on_warnings(self, tmp_path: Path) -> None:
        result = runner.invoke(app, [str(_warning_csv(tmp_path)), "--strict"])

        assert result.exit_code == EXIT_VALIDATION_FAILED

    def test_strict_still_passes_clean_data(self, tmp_path: Path) -> None:
        """--strict escalates warnings; it does not invent failures."""
        result = runner.invoke(app, [str(_clean_csv(tmp_path)), "--strict"])

        assert result.exit_code == EXIT_OK

    def test_strict_does_not_change_failing_data(self, tmp_path: Path) -> None:
        result = runner.invoke(app, [str(_failing_csv(tmp_path)), "--strict"])

        assert result.exit_code == EXIT_VALIDATION_FAILED

    def test_strict_explains_why_it_failed(self, tmp_path: Path) -> None:
        """A build that breaks only because of --strict should say so."""
        result = runner.invoke(app, [str(_warning_csv(tmp_path)), "--strict"])

        assert "strict" in result.output.lower()


class TestCliConfigFile:
    """--config must actually load; it was previously a silent no-op."""

    @staticmethod
    def _config(tmp_path: Path, body: str) -> Path:
        path = tmp_path / "marketcheck.toml"
        path.write_text(body, encoding="utf-8")
        return path

    def test_config_file_changes_behaviour(self, tmp_path: Path) -> None:
        """Disabling a rule through the file must be reflected in the report."""
        data = _clean_csv(tmp_path)
        config = self._config(
            tmp_path, 'disabled_rules = ["numerical.volume_anomaly"]\n'
        )
        result = runner.invoke(app, [str(data), "--config", str(config), "-f", "json"])

        assert result.exit_code == EXIT_OK
        volume_rule = next(
            r
            for r in json.loads(result.stdout)["results"]
            if r["rule_id"] == "numerical.volume_anomaly"
        )
        assert volume_rule["status"] == "skip"
        assert "disabled" in volume_rule["message"].lower()

    def test_config_tuning_value_is_applied(self, tmp_path: Path) -> None:
        """A gap of one bar is reported by default; raising the threshold in the
        file must suppress it."""
        data = _write(
            tmp_path / "gap.csv",
            "2024-01-02 09:30:00,100,101,99,100,1000\n"
            "2024-01-02 09:31:00,100,101,99,100,1000\n"
            "2024-01-02 09:32:00,100,101,99,100,1000\n"
            "2024-01-02 09:34:00,100,101,99,100,1000\n",
        )

        default_run = runner.invoke(app, [str(data), "-f", "json"])
        gap_rule = next(
            r
            for r in json.loads(default_run.stdout)["results"]
            if r["rule_id"] == "temporal.gaps_within_session"
        )
        assert gap_rule["status"] == "warn"

        config = self._config(tmp_path, "gap_min_missing_bars = 2\n")
        tuned = runner.invoke(app, [str(data), "--config", str(config), "-f", "json"])
        gap_rule = next(
            r
            for r in json.loads(tuned.stdout)["results"]
            if r["rule_id"] == "temporal.gaps_within_session"
        )
        assert gap_rule["status"] == "pass"

    def test_cli_strict_overrides_config_file(self, tmp_path: Path) -> None:
        """DD12: CLI flags take precedence, so a one-off --strict can override a
        committed config."""
        data = _warning_csv(tmp_path)
        config = self._config(tmp_path, "max_rows_in_details = 10\n")

        assert runner.invoke(app, [str(data), "--config", str(config)]).exit_code == EXIT_OK
        strict = runner.invoke(app, [str(data), "--config", str(config), "--strict"])
        assert strict.exit_code == EXIT_VALIDATION_FAILED

    def test_bad_config_is_a_tool_error(self, tmp_path: Path) -> None:
        data = _clean_csv(tmp_path)
        config = self._config(tmp_path, "volume_anomly_multiplier = 15.0\n")
        result = runner.invoke(app, [str(data), "--config", str(config)])

        assert result.exit_code == EXIT_TOOL_ERROR
        assert "unknown setting" in result.output

    def test_missing_config_is_a_tool_error(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app, [str(_clean_csv(tmp_path)), "--config", str(tmp_path / "nope.toml")]
        )

        assert result.exit_code == EXIT_TOOL_ERROR

    def test_no_config_flag_uses_defaults(self, tmp_path: Path) -> None:
        result = runner.invoke(app, [str(_clean_csv(tmp_path)), "-f", "json"])

        assert result.exit_code == EXIT_OK
        assert json.loads(result.stdout)["overall_status"] == "pass"
