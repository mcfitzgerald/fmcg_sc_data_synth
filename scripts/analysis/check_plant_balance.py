#!/usr/bin/env python3
"""Check plant production load balance from simulation results."""

import argparse
from pathlib import Path

import pandas as pd


def analyze_plant_load(output_dir: Path) -> int:
    """Analyze plant production load balance."""
    batches_file = output_dir / "batches.parquet"
    if not batches_file.exists():
        print(f"Error: {batches_file} does not exist")
        return 1

    print("Loading batch history...")
    df = pd.read_parquet(batches_file)

    print(f"\n=== TOTAL BATCHES PER PLANT ===")
    print(df['plant_id'].value_counts())

    print(f"\n=== TOTAL VOLUME PER PLANT (Cases) ===")
    vol = df.groupby('plant_id')['quantity'].sum()
    print(vol.apply(lambda x: f"{x:,.0f}"))

    # Check if any plant is consistently idle
    # Group by day and plant
    daily = df.groupby(['day_produced', 'plant_id']).size().unstack(fill_value=0)

    print(f"\n=== AVERAGE BATCHES PER DAY ===")
    print(daily.mean())

    print(f"\n=== DAYS WITH ZERO PRODUCTION ===")
    zero_days = (daily == 0).sum()
    print(zero_days)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Check plant production load balance")
    parser.add_argument(
        "data_dir",
        type=Path,
        nargs="?",
        default=Path("data/output"),
        help="Path to results directory (default: data/output)",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Error: {args.data_dir} does not exist")
        return 1

    return analyze_plant_load(args.data_dir)


if __name__ == "__main__":
    raise SystemExit(main())
