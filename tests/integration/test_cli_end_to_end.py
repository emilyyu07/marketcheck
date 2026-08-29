"""End-to-end CLI integration test."""

from pathlib import Path

from typer.testing import CliRunner

from marketcheck.cli.main import app

runner = CliRunner()


class TestCliEndToEnd:
    def test_validate_csv_text_format(self, sample_csv_path: Path) -> None:
        """CLI produces a text report on a valid CSV without crashing."""
        result = runner.invoke(app, [str(sample_csv_path)])
        assert result.exit_code == 0
        assert "MarketCheck Report" in result.stdout

    def test_validate_csv_json_format(self, sample_csv_path: Path) -> None:
        """CLI produces valid JSON output."""
        result = runner.invoke(app, [str(sample_csv_path), "--format", "json"])
        assert result.exit_code == 0
        # Should be valid JSON containing expected keys
        import json

        data = json.loads(result.stdout)
        assert "overall_status" in data
        assert "results" in data

    def test_validate_parquet(self, sample_parquet_path: Path) -> None:
        """CLI handles Parquet files."""
        result = runner.invoke(app, [str(sample_parquet_path), "--format", "json"])
        assert result.exit_code == 0

    def test_validate_nonexistent_file(self) -> None:
        """CLI exits with error on missing file."""
        result = runner.invoke(app, ["/tmp/nonexistent_file.csv"])
        assert result.exit_code == 1

    def test_validate_with_strict_flag(self, sample_csv_path: Path) -> None:
        """CLI accepts --strict flag without crashing."""
        result = runner.invoke(app, [str(sample_csv_path), "--strict"])
        assert result.exit_code == 0

    def test_validate_with_output_file(self, sample_csv_path: Path, tmp_path: Path) -> None:
        """CLI writes report to file when --output is given."""
        output_file = tmp_path / "report.txt"
        result = runner.invoke(app, [str(sample_csv_path), "--output", str(output_file)])
        assert result.exit_code == 0
        assert output_file.exists()
        assert len(output_file.read_text()) > 0
