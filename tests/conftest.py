"""Shared test fixtures for MarketCheck."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest


@pytest.fixture
def sample_ohlcv_df() -> pl.DataFrame:
    """A minimal valid OHLCV DataFrame (~10 rows) for testing."""
    return pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 2, 9, 30),
                datetime(2024, 1, 2, 9, 31),
                datetime(2024, 1, 2, 9, 32),
                datetime(2024, 1, 2, 9, 33),
                datetime(2024, 1, 2, 9, 34),
                datetime(2024, 1, 2, 9, 35),
                datetime(2024, 1, 2, 9, 36),
                datetime(2024, 1, 2, 9, 37),
                datetime(2024, 1, 2, 9, 38),
                datetime(2024, 1, 2, 9, 39),
            ],
            "open": [100.0, 100.5, 101.0, 100.8, 101.2, 101.5, 101.3, 101.8, 102.0, 101.9],
            "high": [100.8, 101.2, 101.5, 101.3, 101.8, 102.0, 101.9, 102.3, 102.5, 102.2],
            "low": [99.5, 100.2, 100.7, 100.3, 100.9, 101.1, 100.8, 101.4, 101.6, 101.5],
            "close": [100.5, 101.0, 100.8, 101.2, 101.5, 101.3, 101.8, 102.0, 101.9, 102.1],
            "volume": [1000, 1500, 1200, 1300, 1400, 1100, 1250, 1600, 1350, 1450],
        }
    )


@pytest.fixture
def sample_csv_path(tmp_path: Path, sample_ohlcv_df: pl.DataFrame) -> Path:
    """Write sample OHLCV data to a temporary CSV file."""
    path = tmp_path / "test_data.csv"
    sample_ohlcv_df.write_csv(path)
    return path


@pytest.fixture
def sample_parquet_path(tmp_path: Path, sample_ohlcv_df: pl.DataFrame) -> Path:
    """Write sample OHLCV data to a temporary Parquet file."""
    path = tmp_path / "test_data.parquet"
    sample_ohlcv_df.write_parquet(path)
    return path
