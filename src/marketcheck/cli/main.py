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
from marketcheck.reporting.json_report import render_json
from marketcheck.reporting.text_report import render_text

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
        raise typer.Exit(code=1)

    # Load data
    suffix = file.suffix.lower()
    if suffix == ".csv":
        df = load_csv(file)
    elif suffix in (".parquet", ".pq"):
        df = load_parquet(file)
    else:
        typer.echo(f"Error: unsupported file type: {suffix}", err=True)
        raise typer.Exit(code=1)

    # Canonicalize
    dataset = to_canonical(df, source_path=file)

    # Build config
    validation_config = ValidationConfig(
        strict=strict,
        output_format=format,
        output_path=output,
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


if __name__ == "__main__":
    app()
