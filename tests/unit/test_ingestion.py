"""Tests for the ingestion module."""

from pathlib import Path

import polars as pl

from marketcheck.ingestion.canonicalize import to_canonical
from marketcheck.ingestion.loaders import load_csv, load_parquet
from marketcheck.ingestion.schema import REQUIRED_COLUMNS, coerce_dtypes


class TestLoadCsv:
    def test_loads_valid_csv(self, sample_csv_path: Path) -> None:
        df = load_csv(sample_csv_path)
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 10

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        import pytest

        with pytest.raises(FileNotFoundError):
            load_csv(tmp_path / "nonexistent.csv")


class TestLoadParquet:
    def test_loads_valid_parquet(self, sample_parquet_path: Path) -> None:
        df = load_parquet(sample_parquet_path)
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 10


class TestCoerceDtypes:
    def test_coerces_valid_df(self, sample_ohlcv_df: pl.DataFrame) -> None:
        result = coerce_dtypes(sample_ohlcv_df)
        assert all(col in result.columns for col in REQUIRED_COLUMNS)


class TestToCanonical:
    def test_produces_canonical_dataset(self, sample_ohlcv_df: pl.DataFrame) -> None:
        dataset = to_canonical(sample_ohlcv_df, source_path="test.csv")
        assert dataset.row_count == 10
        assert dataset.source_path == "test.csv"
