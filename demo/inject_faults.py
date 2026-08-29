"""Inject specific data quality faults into a clean OHLCV dataset.

Usage:
    python demo/inject_faults.py --input clean.csv --output faulted.csv
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject faults into OHLCV data.")
    parser.add_argument("--input", type=str, required=True, help="Path to clean dataset.")
    parser.add_argument("--output", type=str, default="faulted.csv", help="Output file path.")
    args = parser.parse_args()

    # TODO: Implement fault injection
    print(f"TODO: inject faults from {args.input} → {args.output}")
    sys.exit(0)


if __name__ == "__main__":
    main()
