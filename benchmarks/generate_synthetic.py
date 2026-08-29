"""Generate synthetic OHLCV datasets for benchmarking.

Usage:
    python benchmarks/generate_synthetic.py --rows 1000000 --output data/bench.csv
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic OHLCV data.")
    parser.add_argument("--rows", type=int, default=100_000, help="Number of rows to generate.")
    parser.add_argument("--output", type=str, default="synthetic.csv", help="Output file path.")
    args = parser.parse_args()

    # TODO: Implement synthetic data generation
    print(f"TODO: generate {args.rows:,} rows of synthetic OHLCV data → {args.output}")
    sys.exit(0)


if __name__ == "__main__":
    main()
