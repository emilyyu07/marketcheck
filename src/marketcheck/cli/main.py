"""MarketCheck CLI — validate OHLCV market data files."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from marketcheck.engine.aggregator import aggregate
from marketcheck.engine.runner import run_validation
from marketcheck.ingestion.canonicalize import to_canonical
from marketcheck.ingestion.loaders import load_csv, load_parquet
from marketcheck.models.config import ValidationConfig
from marketcheck.models.config_file import ConfigError, load_config
from marketcheck.models.enums import Status
from marketcheck.reporting.json_report import render_json
from marketcheck.reporting.text_report import render_text

# Exit codes. Validation failure and tool failure are deliberately distinct so a
# CI pipeline can tell "the data is bad" from "the tool could not run". 2 is used
# for tool errors because Typer/Click already exits 2 on usage errors.
EXIT_OK = 0
EXIT_VALIDATION_FAILED = 1
EXIT_TOOL_ERROR = 2

app = typer.Typer(
    name="marketcheck",
    help="Validate historical US-equity OHLCV market data.",
    add_completion=False,
)


@app.command()
def validate(
    file: Annotated[Path, typer.Argument(help="Path to a CSV or Parquet file to validate.")],
    format: Annotated[str, typer.Option("--format", "-f", help="Output format.")] = "text",
    strict: Annotated[bool, typer.Option("--strict", help="Treat warnings as failures.")] = False,
    config: Annotated[
        Optional[Path], typer.Option("--config", "-c", help="Path to config file.")
    ] = None,
    output: Annotated[
        Optional[Path], typer.Option("--output", "-o", help="Write report to file.")
    ] = None,
) -> None:
    """Validate an OHLCV data file and produce a quality report."""
    # Resolve the file
    if not file.exists():
        typer.echo(f"Error: file not found: {file}", err=True)
        raise typer.Exit(code=EXIT_TOOL_ERROR)

    # Load data
    suffix = file.suffix.lower()
    if suffix not in (".csv", ".parquet", ".pq"):
        typer.echo(f"Error: unsupported file type: {suffix}", err=True)
        raise typer.Exit(code=EXIT_TOOL_ERROR)

    # A file that cannot be read or normalised at all is a TOOL error, not a
    # validation failure -- there is no data to render a report about. Note this
    # is a backstop for genuinely malformed input: a file that merely *lacks*
    # columns now loads fine and is reported by structural.missing_columns.
    try:
        if suffix == ".csv":
            df = load_csv(file)
        else:
            df = load_parquet(file)
        dataset = to_canonical(df, source_path=file)
    except Exception as exc:  # noqa: BLE001 - surface any read/parse failure cleanly
        typer.echo(f"Error: could not read {file}: {exc}", err=True)
        raise typer.Exit(code=EXIT_TOOL_ERROR) from exc

    # Build config: file first (validation behaviour), then CLI flags on top.
    # CLI flags take precedence so a one-off `--strict` can override a committed
    # config file.
    if config is not None:
        try:
            validation_config = load_config(config)
        except ConfigError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=EXIT_TOOL_ERROR) from exc
    else:
        validation_config = ValidationConfig()

    validation_config = validation_config.model_copy(
        update={
            "strict": strict,
            "output_format": format,
            "output_path": output,
        }
    )

    # Run validation
    results = run_validation(dataset, config=validation_config)

    # Aggregate
    summary = aggregate(results, dataset)

    # Render report
    if format == "json":
        report = render_json(summary)
    else:
        report = render_text(summary)

    # Output
    if output:
        output.write_text(report, encoding="utf-8")
        typer.echo(f"Report written to {output}")
    else:
        typer.echo(report)

    # Signal the outcome through the exit code so the tool is usable in CI.
    # A WARN is not a failure by default -- warnings cover findings a human must
    # judge (a real trading halt, a genuine 70% biotech drop), so failing the
    # build on them would make the tool unusable on real data. `--strict` is for
    # callers who want any finding at all to break the build.
    if summary.overall_status == Status.FAIL:
        raise typer.Exit(code=EXIT_VALIDATION_FAILED)
    if strict and summary.overall_status == Status.WARN:
        typer.echo(
            "Failing because --strict treats warnings as failures.", err=True
        )
        raise typer.Exit(code=EXIT_VALIDATION_FAILED)


if __name__ == "__main__":
    app()
